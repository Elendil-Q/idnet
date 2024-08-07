import torch
import sys
import os

share_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
sys.path.append(share_path)

import numpy as np
import warp_event


def iwe_cuda(flow, event_list, res):
    """
       Args:
           flow:
           event_list: [y,x,t,p]
           res: [h,w]
           flow_scaling:
           round_idx:

       Returns:
    """
    evt_y = event_list[:, 0].astype(np.float32)
    evt_x = event_list[:, 1].astype(np.float32)
    evt_p = event_list[:, 3].astype(np.uint8)
    evt_t = event_list[:, 2].astype(np.float64)
    optic_flow = flow.astype(np.float32)
    h, w = res
    optic_flow_x = optic_flow[:, :, 0]
    optic_flow_y = optic_flow[:, :, 1]
    iwe = warp_event.motion_compensation_gpu(evt_x, evt_y, evt_p, evt_t, optic_flow_x,optic_flow_y, h, w)
    return iwe


def free_gpu_memory():
    warp_event.free_memory()


if __name__ == "__main__":
    # 生成事件流
    h = 480
    w = 640
    event_num = 10000
    evt_x = np.random.randint(0, w, event_num)
    evt_y = np.random.randint(0, h, event_num)
    evt_p = np.random.randint(0, 2, event_num)
    evt_t = np.linspace(0, 1, event_num)

    # 生成光流，float
    optic_flow = np.random.randn(h, w, 2).astype(np.float32)

    iwe = iwe_cuda(optic_flow, np.stack([evt_y, evt_x, evt_t, evt_p], axis=1), [h, w])

    print(1)

    warp_event.free_memory()
