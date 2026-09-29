"""Tiny unbuffered-friendly training stand-in for local smoke tests."""

import time

for step in range(3):
    print(f"step={step} loss={1.0 / (step + 1):.6f}", flush=True)
    time.sleep(0.05)

