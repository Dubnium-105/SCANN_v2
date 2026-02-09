"""Simple test to verify logging implementation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scann.logger_config import setup_logging, get_logger

print("=" * 60)
print("Testing Logging System")
print("=" * 60)

# Test 1: Initialize logging
print("\n[Test 1] Initialize logging...")
test_log_file = Path("test_verify.log")
logger = setup_logging(log_file=test_log_file)
print(f"✓ Logger created: {logger.name}")
print(f"✓ Log file path: {test_log_file}")

# Test 2: Write different log levels
print("\n[Test 2] Write different log levels...")
logger.debug("This is a DEBUG message (should NOT appear)")
logger.info("This is an INFO message")
logger.warning("This is a WARNING message")
logger.error("This is an ERROR message")
print("✓ Messages written")

# Test 3: Check log file
print("\n[Test 3] Check log file...")
if test_log_file.exists():
    print("✓ Log file exists")
    with open(test_log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✓ Log file size: {len(content)} bytes")
    print("\n--- Log file content ---")
    print(content)
    print("--- End of log file ---")

    # Verify content
    assert "INFO message" in content, "INFO message not found"
    assert "WARNING message" in content, "WARNING message not found"
    assert "ERROR message" in content, "ERROR message not found"
    assert "DEBUG message" not in content, "DEBUG message should not appear"
    print("✓ Log content verified")
else:
    print("✗ Log file does not exist")
    sys.exit(1)

# Test 4: Get logger
print("\n[Test 4] Get named logger...")
test_logger = get_logger("test.module")
print(f"✓ Logger created: {test_logger.name}")
test_logger.info("Message from test.logger")

# Cleanup
print("\n[Cleanup] Deleting test log file...")
if test_log_file.exists():
    test_log_file.unlink()
    print("✓ Test log file deleted")

print("\n" + "=" * 60)
print("All tests PASSED!")
print("=" * 60)
