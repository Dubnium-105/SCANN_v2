"""Manual smoke test for the shared logging configuration."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scann.logger_config import close_logging, get_logger, setup_logging


def main() -> int:
    log_file = PROJECT_ROOT / "logs" / "logging_smoke.log"

    print("=" * 60)
    print("Testing logging system")
    print("=" * 60)

    logger = setup_logging(log_file=log_file)
    print(f"Logger created: {logger.name}")
    print(f"Log file path: {log_file}")

    logger.debug("This DEBUG message should not be written.")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    print("Messages written.")

    if not log_file.exists():
        print("Log file was not created.")
        return 1

    content = log_file.read_text(encoding="utf-8")
    print("\n--- Log file content ---")
    print(content)
    print("--- End of log file ---")

    assert "INFO message" in content, "INFO message not found"
    assert "WARNING message" in content, "WARNING message not found"
    assert "ERROR message" in content, "ERROR message not found"
    assert "DEBUG message" not in content, "DEBUG message should not appear"

    named_logger = get_logger("diagnostics.logging_smoke")
    print(f"Named logger created: {named_logger.name}")
    named_logger.info("Message from named logger")

    close_logging()
    log_file.unlink(missing_ok=True)
    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
