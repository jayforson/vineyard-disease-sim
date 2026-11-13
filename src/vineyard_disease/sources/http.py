"""Minimal stdlib HTTP helpers: whole-file GET and seekable range reads.

``HttpRangeReader`` lets :mod:`zipfile` read a handful of members of a large
remote archive with HTTP ``Range`` requests, so preparing a bounded subset
never downloads the whole archive. Every byte fetched is charged to a
:class:`ByteBudget`, which aborts once the configured bound is exceeded.
"""

from __future__ import annotations

import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .. import __version__
from ..errors import PipelineError, SourceUnavailableError

USER_AGENT = f"vineyard-disease/{__version__} (+python-urllib)"


@dataclass
class Response:
    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str


Transport = Callable[[str, Mapping[str, str], float], Response]


SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urllib.parse.urlsplit(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    return parts.scheme.lower(), (parts.hostname or "").lower(), port


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never carry credentials to a different origin.

    urllib's default handler copies every request header (including
    ``Authorization``) to the redirect target, even on another host.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and _origin(req.full_url) != _origin(new.full_url):
            for name in list(new.headers):
                if name.lower() in SENSITIVE_HEADERS:
                    del new.headers[name]
            for name in list(new.unredirected_hdrs):
                if name.lower() in SENSITIVE_HEADERS:
                    del new.unredirected_hdrs[name]
        return new


def build_opener() -> urllib.request.OpenerDirector:
    """Opener with the safe redirect handler (proxy env vars still honored)."""
    return urllib.request.build_opener(SafeRedirectHandler)


def urllib_transport(url: str, headers: Mapping[str, str], timeout: float) -> Response:
    """Default transport: http(s) only, credential-stripping redirects."""
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        raise SourceUnavailableError(f"refusing non-HTTP URL {_redact(url)}")
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with build_opener().open(req, timeout=timeout) as resp:
            return Response(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read(), resp.geturl())
    except urllib.error.HTTPError as exc:
        body = exc.read() if exc.fp else b""
        return Response(exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, body, url)


class ByteBudget:
    def __init__(self, max_bytes: int | None):
        self.max_bytes = max_bytes
        self.used = 0

    def charge(self, n: int) -> None:
        self.used += n
        if self.max_bytes is not None and self.used > self.max_bytes:
            raise PipelineError(
                f"download budget exceeded: {self.used} bytes > {self.max_bytes} bytes; "
                "lower --max-per-class or raise --max-download-mb"
            )


class HttpClient:
    def __init__(
        self,
        transport: Transport | None = None,
        *,
        timeout: float = 60.0,
        retries: int = 3,
        backoff_s: float = 1.0,
        budget: ByteBudget | None = None,
    ):
        # The dataset is public: no credentials (e.g. ambient HF_TOKEN) are ever sent.
        self.transport = transport or urllib_transport
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff_s = backoff_s
        self.budget = budget or ByteBudget(None)

    def request(self, url: str, headers: Mapping[str, str] | None = None) -> Response:
        hdrs = {"User-Agent": USER_AGENT}
        hdrs.update(headers or {})
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.transport(url, hdrs, self.timeout)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                last_exc = exc
            else:
                if resp.status < 500 and resp.status != 429:
                    self.budget.charge(len(resp.body))
                    return resp
                last_exc = SourceUnavailableError(f"HTTP {resp.status} from {_redact(url)}")
            if attempt + 1 < self.retries:
                time.sleep(self.backoff_s * (2**attempt))
        raise SourceUnavailableError(
            f"could not download {_redact(url)}: {last_exc}. No data was synthesized; "
            "retry later or supply local copies of the source files."
        )

    def get_bytes(self, url: str) -> bytes:
        resp = self.request(url)
        if resp.status != 200:
            raise SourceUnavailableError(f"HTTP {resp.status} for {_redact(url)}; no data was synthesized")
        return resp.body


def _redact(url: str) -> str:
    """Drop query strings (signed CDN URLs) from messages."""
    return url.split("?", 1)[0]


_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)")


class HttpRangeReader(io.RawIOBase):
    """Read-only seekable view of a remote file using HTTP Range requests."""

    def __init__(self, url: str, client: HttpClient, *, block_size: int = 64 * 1024, max_blocks: int = 64):
        super().__init__()
        self.url = url
        self.client = client
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self.requests = 0
        self.final_url, self.size, self.headers = self._resolve()

    # -- remote access -------------------------------------------------
    def _resolve(self) -> tuple[str, int, Mapping[str, str]]:
        resp = self.client.request(self.url, {"Range": "bytes=0-0"})
        self.requests += 1
        if resp.status != 206:
            raise SourceUnavailableError(
                f"server did not honor a Range request for {_redact(self.url)} (HTTP {resp.status}); "
                "supply a local copy of the archive instead"
            )
        m = _CONTENT_RANGE.match(resp.headers.get("content-range", ""))
        if not m or m.group(3) == "*":
            raise SourceUnavailableError(f"missing Content-Range for {_redact(self.url)}")
        return resp.url, int(m.group(3)), resp.headers

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch bytes [start, end) (end exclusive)."""
        if end <= start:
            return b""
        for attempt in range(2):
            resp = self.client.request(self.final_url, {"Range": f"bytes={start}-{end - 1}"})
            self.requests += 1
            if resp.status in (401, 403, 410) and attempt == 0:
                self.final_url, _, _ = self._resolve()  # signed URL expired
                continue
            if resp.status != 206 or len(resp.body) != end - start:
                raise SourceUnavailableError(
                    f"bad range response HTTP {resp.status} ({len(resp.body)} of {end - start} bytes) "
                    f"for {_redact(self.url)}"
                )
            return resp.body
        raise SourceUnavailableError(f"range request failed for {_redact(self.url)}")  # pragma: no cover

    def _block(self, idx: int) -> bytes:
        if idx in self._blocks:
            self._blocks.move_to_end(idx)
            return self._blocks[idx]
        start = idx * self.block_size
        data = self._fetch(start, min(self.size, start + self.block_size))
        self._blocks[idx] = data
        while len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return data

    # -- io.RawIOBase ----------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.size - self._pos))
        if n == 0:
            return 0
        start, end = self._pos, self._pos + n
        if n > self.block_size * 4:
            data = self._fetch(start, end)  # large read (e.g. central directory): no caching
        else:
            parts = []
            first, last = start // self.block_size, (end - 1) // self.block_size
            for idx in range(first, last + 1):
                blk = self._block(idx)
                lo = max(start, idx * self.block_size) - idx * self.block_size
                hi = min(end, (idx + 1) * self.block_size) - idx * self.block_size
                parts.append(blk[lo:hi])
            data = b"".join(parts)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)
