from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="model.pt", data="config.yaml", imgsz=512, half=False, device="cpu")