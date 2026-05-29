"""pygalago.tools — command-line entry points (Phase 7)."""

__all__ = ["main"]


def main() -> None:
    from pygalago.tools.cli import main as _main
    _main()
