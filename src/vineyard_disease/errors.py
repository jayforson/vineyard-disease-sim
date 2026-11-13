"""Exception types with stable CLI exit codes."""

from __future__ import annotations


class PipelineError(Exception):
    """Base error; ``exit_code`` is returned by the CLI."""

    exit_code = 1


class SourceUnavailableError(PipelineError):
    """A data source could not be read (network, missing file, bad hash).

    The pipeline never substitutes synthetic data when this happens.
    """

    exit_code = 3


class MissingDependencyError(PipelineError):
    """An optional dependency needed by the requested command is missing."""

    exit_code = 4

    def __init__(self, package: str, purpose: str, hint: str | None = None):
        msg = f"'{package}' is required for {purpose} but is not installed."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.package = package


class UnsupportedModelError(PipelineError):
    """The classifier does not cover the required grape classes."""

    exit_code = 5


class InputError(PipelineError):
    """Invalid user input (paths, manifests, configuration)."""

    exit_code = 2


def require(module: str, purpose: str, hint: str | None = None):
    """Import ``module`` lazily, raising :class:`MissingDependencyError`."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        package = module.split(".")[0]
        raise MissingDependencyError(package, purpose, hint) from exc
