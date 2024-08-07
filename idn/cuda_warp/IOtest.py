import torch
import sys
import os

share_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
sys.path.append(share_path)

import numpy as np
import warp_event

# 生成事件流
h = 480
w = 640
event_num = 10000
evt_x = np.random.randint(-10, w+10, event_num)
evt_y = np.random.randint(-10, h+10, event_num)
evt_p = np.random.randint(0, 2, event_num)
evt_t = np.linspace(0, 1, event_num)

# 生成光流，float
optic_flow = np.random.randn(h, w, 2).astype(np.float32)

iwe = warp_event.motion_compensation_gpu(evt_x, evt_y, evt_p, evt_t, optic_flow, h, w)

print(1)

cuda_warp.free_memory()
