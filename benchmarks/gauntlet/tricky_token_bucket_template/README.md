# Tricky Token Bucket Gauntlet

This benchmark has subtle token bucket bugs around boundary checks and refill precision.

Goal: make all unit tests pass.

Run tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
