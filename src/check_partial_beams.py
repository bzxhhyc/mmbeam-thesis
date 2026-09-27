"""用当前已算的网格 CIR 提前检查 beam_index 多样性（不等全量跑完）"""
import os
import numpy as np

cache = os.path.join(os.path.dirname(__file__), "..", "data", "grid_cir_100.npz")
if not os.path.exists(cache):
    print("缓存还没生成（Phase1 未跑完）")
    # 临时方案：读 log 看进度，手动估算
    sys.exit()

z = np.load(cache)
beam_indices = []
for k in z.files:
    powers = z[k]
    beam_idx = np.argmax(powers) + 1
    beam_indices.append(beam_idx)

unique, counts = np.unique(beam_indices, return_counts=True)
print(f"已算网格点数: {len(z.files)}")
print(f"unique beam_index: {len(unique)}/64")
print(f"分布: {dict(zip(unique.tolist(), counts.tolist()))}")
