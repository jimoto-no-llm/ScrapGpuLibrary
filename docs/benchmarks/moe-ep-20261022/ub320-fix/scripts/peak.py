#!/usr/bin/env python3
"""peak VRAM use of RX 6400 (card1) and WX 2100 (card5) and GTT (host-visible system memory) until SIGTERM. usage: peak.py OUT.txt"""
import sys, time, signal
stop = False
def h(*a):
    global stop; stop = True
signal.signal(signal.SIGTERM, h); signal.signal(signal.SIGINT, h)
def rd(p):
    try: return int(open(p).read())
    except Exception: return 0
F = {"rx_vram": "/sys/class/drm/card1/device/mem_info_vram_used", "rx_gtt": "/sys/class/drm/card1/device/mem_info_gtt_used",
     "wx_vram": "/sys/class/drm/card5/device/mem_info_vram_used", "wx_gtt": "/sys/class/drm/card5/device/mem_info_gtt_used"}
base = {k: rd(p) for k, p in F.items()}
mx = dict(base)
while not stop:
    for k, p in F.items():
        v = rd(p)
        if v > mx[k]: mx[k] = v
    time.sleep(0.1)
with open(sys.argv[1], "w") as f:
    f.write(" ".join("%s_base=%.0fMiB %s_peak=%.0fMiB" % (k, base[k] / 1048576, k, mx[k] / 1048576) for k in F) + "\n")
