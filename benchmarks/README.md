# Benchmarks

These benchmarks were run like this:
```
for W in {1..8}; do python tests/test_load.py -u http://server:8000 -i tests/em1.png -w $W -r 10; done >> gpu_rtx8000_direct.txt
```

