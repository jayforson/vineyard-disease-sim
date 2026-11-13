"""Security/robustness hardening: real urllib redirects, size bounds, bombs, empty results."""

import json
import os
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(__file__))

import pytest

pytest.importorskip("PIL")

from test_pipeline_support import REV, FakeHub, MockClassifier, build_fake_dataset, client_for, png_bytes

from vineyard_disease.errors import InputError, SourceUnavailableError
from vineyard_disease.gating import GateConfig
from vineyard_disease.manifest import read_manifest
from vineyard_disease.predict import collect_inputs, predict_iter
from vineyard_disease.prepare import run_prepare
from vineyard_disease.quality import inspect_image_bytes
from vineyard_disease.sources.http import HttpClient, HttpRangeReader, urllib_transport
from vineyard_disease.sources.local_dir import LocalClassDirs
from vineyard_disease.sources.plantvillage_hf import PlantVillageHF


# -- real urllib redirect semantics against local servers --------------------------
class _Server:
    """Tiny local HTTP server: optional redirect, Range support, header capture."""

    def __init__(self, body=b"", redirect_to=None):
        self.body = body
        self.redirect_to = redirect_to
        self.seen: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen.append({k.lower(): v for k, v in self.headers.items()} | {"_path": self.path})
                location = outer.redirect_to(self.path) if outer.redirect_to else None
                if location:
                    self.send_response(302)
                    self.send_header("Location", location)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                data = outer.body
                if rng:
                    start, end = (int(v) for v in rng.split("=", 1)[1].split("-"))
                    end = min(end, len(data) - 1)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    data = data[start : end + 1]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def no_proxy(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_cross_origin_redirect_strips_credentials(no_proxy):
    target = _Server(body=b"0123456789")
    origin = _Server(redirect_to=lambda path: target.url + path)  # different port = different origin
    try:
        resp = urllib_transport(
            origin.url + "/file",
            {"Authorization": "Bearer secret", "Cookie": "c=1", "Proxy-Authorization": "p", "Range": "bytes=2-4"},
            10,
        )
    finally:
        origin.close()
        target.close()
    assert origin.seen[0]["authorization"] == "Bearer secret"  # first hop is the requested origin
    hop = target.seen[0]
    assert "authorization" not in hop and "cookie" not in hop and "proxy-authorization" not in hop
    assert hop["range"] == "bytes=2-4"  # non-sensitive headers survive the redirect
    assert resp.status == 206 and resp.body == b"234" and resp.url.startswith(target.url)


def test_same_origin_redirect_is_followed(no_proxy):
    server = _Server(body=b"abcdef")
    server.redirect_to = lambda path: server.url + "/final" if path == "/start" else None
    try:
        resp = urllib_transport(server.url + "/start", {"Authorization": "Bearer x", "Range": "bytes=0-1"}, 10)
    finally:
        server.close()
    assert [h["_path"] for h in server.seen] == ["/start", "/final"]
    assert server.seen[1]["authorization"] == "Bearer x"  # same origin: unchanged urllib behavior
    assert resp.status == 206 and resp.body == b"ab" and resp.url.endswith("/final")


def test_range_reader_over_real_urllib_redirect(no_proxy, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")  # must never be sent
    files, _ = build_fake_dataset()
    cdn = _Server(body=files["data.zip"])
    hub = _Server(redirect_to=lambda path: cdn.url + "/blob?sig=1")
    try:
        reader = HttpRangeReader(hub.url + "/data.zip", HttpClient(backoff_s=0), block_size=2048)
        with zipfile.ZipFile(reader) as zf:
            data = zf.read("raw/color/Grape___Black_rot/a1___FAM_B.Rot 0001.JPG")
    finally:
        hub.close()
        cdn.close()
    assert data[:4] == b"\x89PNG"
    assert all("authorization" not in h for h in hub.seen + cdn.seen)
    assert len(hub.seen) == 1  # later range reads go straight to the resolved URL
    assert all("range" in h for h in cdn.seen)


def test_non_http_urls_are_refused():
    with pytest.raises(SourceUnavailableError, match="non-HTTP"):
        urllib_transport("file:///etc/hostname", {}, 1)


# -- archive member size bound ----------------------------------------------------
def test_oversized_zip_member_rejected_before_read(tmp_path):
    big = "raw/color/Grape___healthy/zz___Big 0001.JPG"
    files, _ = build_fake_dataset(extra_members={big: os.urandom(100_000)})
    files["splits/color_test.txt"] += (big + "\n").encode()
    src = PlantVillageHF(client_for(FakeHub(files)), cache_dir=tmp_path / "c", revision=REV, max_member_bytes=50_000)
    out = tmp_path / "pv"
    run_prepare(src, out, max_per_class=10)
    prov = json.loads((out / "provenance.json").read_text())
    assert [r["reasons"] for r in prov["rejected"]] == [["member_too_large"]]
    assert all("Big" not in r["source"]["path"] for r in read_manifest(out / "manifest.jsonl"))


def test_zero_surviving_records_is_an_error_and_keeps_previous_output(tmp_path):
    files, _ = build_fake_dataset()
    out = tmp_path / "pv"
    run_prepare(
        PlantVillageHF(client_for(FakeHub(files)), cache_dir=tmp_path / "c", revision=REV), out, max_per_class=1
    )
    before = (out / "manifest.jsonl").read_bytes()
    tiny = PlantVillageHF(client_for(FakeHub(files)), cache_dir=tmp_path / "c", revision=REV, max_member_bytes=10)
    with pytest.raises(InputError, match="no images survived"):
        run_prepare(tiny, out, max_per_class=1, overwrite=True)
    assert (out / "manifest.jsonl").read_bytes() == before
    assert (out / "images").is_dir() and not (out / "images.partial").exists()
    with pytest.raises(InputError, match="no manifest was written"):
        run_prepare(tiny, tmp_path / "fresh", max_per_class=1)
    assert not (tmp_path / "fresh" / "manifest.jsonl").exists()


def test_local_oversized_file_rejected(tmp_path, monkeypatch):
    from vineyard_disease.sources import local_dir

    (tmp_path / "in" / "esca").mkdir(parents=True)
    (tmp_path / "in" / "esca" / "a.png").write_bytes(png_bytes())
    monkeypatch.setattr(local_dir, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(InputError, match="file_too_large"):
        run_prepare(LocalClassDirs(tmp_path / "in"), tmp_path / "out")


# -- decompression bombs ------------------------------------------------------------
@pytest.mark.parametrize("max_pixels", [1000, 3000])  # 64x64=4096: error (>2x) and warning (>1x) ranges
def test_decompression_bomb_is_a_hard_quality_failure(monkeypatch, max_pixels):
    from PIL import Image

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)
    report = inspect_image_bytes(png_bytes(size=(64, 64)))
    assert not report.ok and report.issues == ["decompression_bomb"]


@pytest.mark.parametrize("max_pixels", [1000, 3000])
def test_decompression_bomb_is_input_error_in_predict(tmp_path, monkeypatch, max_pixels):
    from PIL import Image

    p = tmp_path / "big.png"
    p.write_bytes(png_bytes(size=(64, 64)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)
    [row] = predict_iter(collect_inputs([str(p)]), MockClassifier(), GateConfig())
    assert row["status"] == "input_error" and "DecompressionBomb" in row["error"]


def test_prepare_with_only_bombs_fails(tmp_path, monkeypatch):
    from PIL import Image

    (tmp_path / "in" / "esca").mkdir(parents=True)
    (tmp_path / "in" / "esca" / "a.png").write_bytes(png_bytes(size=(64, 64)))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    with pytest.raises(InputError, match="decompression_bomb"):
        run_prepare(LocalClassDirs(tmp_path / "in"), tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.jsonl").exists()
