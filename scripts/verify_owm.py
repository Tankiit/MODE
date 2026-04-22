import numpy as np

t = len(np.memmap("./data/owm/train.bin", dtype=np.uint16, mode="r"))
v = len(np.memmap("./data/owm/val.bin", dtype=np.uint16, mode="r"))
print(f"train: {t/1e6:.1f}M  val: {v/1e6:.1f}M  total: {(t+v)/1e9:.3f}B")
