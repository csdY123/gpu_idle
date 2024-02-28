import os
import subprocess
import time
import logging
import random
from multiprocessing import Process

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(levelname)s] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S')
print = logging.info


def get_gpu_info(visible_gpus=None):
    res = subprocess.getoutput('nvidia-smi')
    res = res.split('\n')

    gpu_info = []
    for i, s in enumerate(res):
        flag = True
        for x in ['%', 'W', 'C', 'MiB']:
            flag = flag and (x in s)
        if not flag:
            continue

        id = len(gpu_info)
        info = s.split(' ')
        #print(info)

        pwr_use, pwr_total = [float(x.split('W')[0]) for x in info if 'W' in x]
        mem_use, mem_total = [float(x.split('MiB')[0]) for x in info if 'MiB' in x]
        gpu_fan, gpu_util = [float(x.split('%')[0].split('|')[-1]) for x in info if '%' in x]

        if visible_gpus is None or int(id) in visible_gpus:
            gpu_info.append({'id': id, 'mem_use': mem_use, 'mem_total': mem_total,
                             'pwr_use': pwr_use, 'pwr_total': pwr_total, 'gpu_util': gpu_util})
    return gpu_info


def is_gpu_all_idle():
    gpu_info = get_gpu_info()
    # print(gpu_info)
    is_idle = [x['mem_use'] < 200 or x['gpu_util'] <= 2 for x in gpu_info]
    return all(is_idle)


def is_gpu_idle(device_id):
    gpu_info = get_gpu_info()[device_id]
    # print(gpu_info)
    # mem<200M or gpu_util<2%
    is_idle = gpu_info['mem_use'] < 200 or gpu_info['gpu_util'] <= 2
    return is_idle


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = nn.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = nn.Linear(dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, dtype=dtype, device=device),
                                    nn.Dropout(dropout))

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out)


@torch.no_grad()
def use_gpu(dur=5, device_id=0):
    B = 128
    T = 128
    D = 320
    B = random.randint(128, 2560)
    device = torch.device(f'cuda:{device_id}')

    SA = SelfAttention(D, device=device)
    x = torch.rand((B, T, D), device=device)
    t0 = time.time()
    while time.time() - t0 < dur:
        y = SA(x)
        # print(y.shape)
    return y


def run_in_process(dur, device_id):
    p = Process(target=use_gpu, args=(dur, device_id))
    p.start()
    p.join()  # Wait for the process to finish


def MonitorProcess(device_id, dur_run=4, dur_busy_sleep=300):
    print(f'start process-{device_id}')
    while True:
        try:
            is_idle = is_gpu_idle(device_id)
            if 0 or is_idle:
                # print(f'gpu-{device_id} is idle')
                dur_run = random.randint(3, 6)
                p = Process(target=use_gpu, args=(dur_run, device_id))
                p.start()
                p.join()  # Wait for the process to finish
                time.sleep(random.randint(0, 1))
            else:
                print(f'gpu-{device_id} is busy, wait {dur_busy_sleep}s')
                time.sleep(dur_busy_sleep)  # seconds
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print('main start ...')
    gpu_info = get_gpu_info()
    print(gpu_info)
    num_gpus = len(gpu_info)
    for device_id in range(0, num_gpus):
        p = Process(target=MonitorProcess, args=(device_id,))
        p.start()
    print('main exit.')
