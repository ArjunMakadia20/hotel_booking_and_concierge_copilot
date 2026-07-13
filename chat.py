"""
Entry point for the interactive cancellation-prediction chat tool.

Run:
    python chat.py

Thin wrapper: configures logging and hands off to :func:`src.chat_interface.run_chat`.
"""

from __future__ import annotations

import logging

from src.chat_interface import run_chat


def main() -> None:
    """Configure logging and start the interactive prediction loop."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_chat()


if __name__ == "__main__":
    main()
