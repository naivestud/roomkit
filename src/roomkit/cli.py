"""Command line interface for roomkit."""

from __future__ import annotations

import argparse

from roomkit import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="roomkit",
        description="Speech enhancement and room-acoustics toolkit.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
