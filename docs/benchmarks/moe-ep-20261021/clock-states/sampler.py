#!/usr/bin/env python3
"""20 ms (SAMPLE_MS) sampler of RX 6400 (card1) / WX 2100 (card5) power state + CPU frequency. usage: sampler.py OUT.csv  (stops on SIGTERM/SIGINT)"""
import sys, time, signal, glob, re
import os
IV = float(os.environ.get('SAMPLE_MS', '20')) / 1000   # sampling interval
ONLY = os.environ.get('SAMPLE_COLS')                      # regex: keep only matching columns
stop = False
def h(*a):
    global stop; stop = True
signal.signal(signal.SIGTERM, h); signal.signal(signal.SIGINT, h)
def rd(p):
    try:
        with open(p) as f: return f.read()
    except Exception: return ""
def star(p):  # active line of pp_dpm_*: "idx:value" of the line marked '*'
    r = []
    for l in rd(p).splitlines():
        if '*' in l:
            r.append(l.replace('*', '').strip().replace(' ', '').replace(',', '/'))
    return '+'.join(r)
def n(p):
    s = rd(p).strip()
    return s if s else ""
cards = {'rx': 'card1', 'wx': 'card5'}
hw = {k: glob.glob(f'/sys/class/drm/{c}/device/hwmon/hwmon*')[0] for k, c in cards.items()}
import struct
GM = struct.Struct('<4x6H3HHQQ7H7HIHHHHIIQ4H'.replace('Q4H', '4HQ') + '3HHQ')   # gpu_metrics v1_3 (RX 6400), 120 bytes
GMCOLS = ['edge', 'hot', 'mem', 'vrgfx', 'vrsoc', 'vrmem', 'act_gfx', 'act_umc', 'act_mm', 'pwr', 'energy', 'clkctr', 'a_gfx', 'a_soc', 'a_uclk', 'a_v0', 'a_d0', 'a_v1', 'a_d1',
          'c_gfx', 'c_soc', 'c_uclk', 'c_v0', 'c_d0', 'c_v1', 'c_d1', 'thr', 'fan', 'lw', 'ls', 'pad', 'acc_gfx', 'acc_mem', 'h0', 'h1', 'h2', 'h3', 'fwts', 'v_soc', 'v_gfx', 'v_mem', 'pad1', 'indep_thr']
GMKEEP = ('act_gfx', 'pwr', 'a_gfx', 'c_gfx', 'c_uclk', 'thr', 'v_gfx', 'hot', 'indep_thr')
cols = ['t']
files = []  # (col, kind, path)
for k, c in cards.items():
    d = f'/sys/class/drm/{c}/device'
    for name in ['sclk', 'mclk', 'fclk', 'socclk', 'pcie']:
        files.append((f'{k}_{name}', 's', f'{d}/pp_dpm_{name}'))
    files += [(f'{k}_lspeed', 'n', f'{d}/current_link_speed'), (f'{k}_lwidth', 'n', f'{d}/current_link_width'),
              (f'{k}_busy', 'n', f'{d}/gpu_busy_percent'), (f'{k}_membusy', 'n', f'{d}/mem_busy_percent'),
              (f'{k}_temp', 'n', f'{hw[k]}/temp1_input'), (f'{k}_pwr', 'n', f'{hw[k]}/power1_average'),
              (f'{k}_freq1', 'n', f'{hw[k]}/freq1_input'),
              (f'{k}_vram', 'n', f'{d}/mem_info_vram_used'), (f'{k}_gtt', 'n', f'{d}/mem_info_gtt_used')]
for z in sorted(glob.glob('/sys/class/thermal/thermal_zone*')):
    if rd(z + '/type').strip() in ('x86_pkg_temp', 'acpi-0', 'acpitz'):
        files.append(('cpu_' + rd(z + '/type').strip() + z[-1], 'n', z + '/temp'))
files.append(('rx_gm', 'g', '/sys/class/drm/card1/device/gpu_metrics'))
for i in (0, 1):
    files.append((f'cpu{i}_khz', 'n', f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq'))
if ONLY:
    import re as _re
    files = [f for f in files if _re.search(ONLY, f[0])]
cols += [f[0] if f[1] != 'g' else None for f in files]
cols = [c for c in cols if c] + ['rx_m_' + k for k in GMKEEP] if any(f[1] == 'g' for f in files) else [c for c in cols if c]
out = open(sys.argv[1], 'w')
out.write(','.join(cols) + '\n')
t0 = time.time(); nxt = t0
while not stop:
    row = [f'{time.time()-t0:.3f}']
    gm = None
    for c, k, p in files:
        if k == 'g':
            try:
                with open(p, 'rb') as gf: gm = dict(zip(GMCOLS, GM.unpack(gf.read(120))))
            except Exception: gm = None
            continue
        row.append(star(p) if k == 's' else n(p))
    if any(f[1] == 'g' for f in files):
        row += [str(gm[k]) if gm else '' for k in GMKEEP]
    out.write(','.join(row) + '\n')
    nxt += IV
    d = nxt - time.time()
    if d > 0: time.sleep(d)
    else: nxt = time.time()
out.close()
