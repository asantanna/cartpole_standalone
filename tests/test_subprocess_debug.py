#!/usr/bin/env python3
"""
Debug tests for subprocess decorator.
"""
import pytest
import subprocess
import sys
import os

# Test that we can run Isaac Gym in a subprocess directly
def test_direct_subprocess():
    """Test Isaac Gym in subprocess without decorator."""
    script = """
import sys
import os
print("Python executable:", sys.executable)
print("Python path:", sys.path[:3])
print("Current directory:", os.getcwd())

# Try to import Isaac Gym
try:
    import isaacgym
    print("Isaac Gym imported successfully")
except Exception as e:
    print(f"Failed to import Isaac Gym: {e}")
    sys.exit(1)

# Try to create environment
try:
    from src.cartpole import make_env
    print("Imported make_env")
    env = make_env(headless=True)
    print("Environment created successfully")
    print(f"Environment type: {type(env)}")
except Exception as e:
    print(f"Failed to create environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    proc = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        cwd=os.getcwd()
    )
    
    print("STDOUT:")
    print(proc.stdout)
    print("\nSTDERR:")
    print(proc.stderr)
    print(f"\nReturn code: {proc.returncode}")
    
    assert proc.returncode == 0, f"Subprocess failed with return code {proc.returncode}"


if __name__ == "__main__":
    test_direct_subprocess()