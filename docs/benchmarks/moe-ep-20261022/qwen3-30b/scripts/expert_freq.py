#!/usr/bin/env python3 -I
"""Per-layer expert activation counts from a llama-imatrix GGUF file (no dependencies).
usage: expert_freq.py imatrix.gguf [--json out.json]
Reads the tensors named blk.N.ffn_{up,gate,down}_exps.weight.counts (F32, one count per expert).
Prints: per layer the share of the top 8 / 16 / 32 experts (of 128), the normalized entropy, and the
cumulative share by expert rank (mean over layers), plus the Jaccard overlap of hot sets is left to compare.py"""
import struct, sys, json, math

def read_gguf(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver, = struct.unpack('<I', f.read(4))
    nt, nkv = struct.unpack('<QQ', f.read(16))
    def rstr():
        n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode()
    def rval(t):
        sz = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if t in sz:
            fmt = '<' + sz[t]; n = struct.calcsize(fmt); return struct.unpack(fmt, f.read(n))[0]
        if t == 8: return rstr()
        if t == 9:
            et, = struct.unpack('<I', f.read(4)); n, = struct.unpack('<Q', f.read(8))
            return [rval(et) for _ in range(n)]
        raise ValueError(t)
    kv = {}
    for _ in range(nkv):
        k = rstr(); t, = struct.unpack('<I', f.read(4)); kv[k] = rval(t)
    infos = []
    for _ in range(nt):
        name = rstr(); nd, = struct.unpack('<I', f.read(4))
        ne = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        ty, off = struct.unpack('<IQ', f.read(12))
        infos.append((name, ne, ty, off))
    align = kv.get('general.alignment', 32)
    pos = f.tell(); base = (pos + align - 1) // align * align
    out = {}
    for name, ne, ty, off in infos:
        if name.endswith('.counts') and ty == 0:
            n = 1
            for d in ne: n *= d
            f.seek(base + off); out[name] = list(struct.unpack('<%df' % n, f.read(4 * n)))
    return out

def main():
    path = sys.argv[1]
    t = read_gguf(path)
    layers = {}
    for name, v in t.items():
        parts = name.split('.')
        if parts[0] == 'blk' and 'ffn_down_exps' in name:
            layers[int(parts[1])] = v
    L = sorted(layers)
    ne = len(layers[L[0]])
    print('file', path, 'layers', len(L), 'experts', ne)
    # counts of ffn_down_exps = number of (token, slot) pairs routed to each expert
    cum_mean = [0.0] * ne
    rows = []
    for l in L:
        c = sorted(layers[l], reverse=True); tot = sum(c)
        cum = []; s = 0
        for x in c: s += x; cum.append(s / tot)
        for i in range(ne): cum_mean[i] += cum[i] / len(L)
        p = [x / tot for x in layers[l] if x > 0]
        H = -sum(x * math.log(x) for x in p) / math.log(ne)
        rows.append((l, tot, cum[7], cum[15], cum[31], H, sum(1 for x in layers[l] if x == 0)))
    print('layer  routed  top8   top16  top32  H/Hmax  unused')
    for r in rows: print('%5d %7d %6.3f %6.3f %6.3f %7.3f %5d' % r)
    print('mean over layers: top8 %.3f top16 %.3f top32 %.3f top64 %.3f (uniform: %.3f %.3f %.3f %.3f)' % (
        cum_mean[7], cum_mean[15], cum_mean[31], cum_mean[63], 8/ne, 16/ne, 32/ne, 64/ne))
    if '--json' in sys.argv:
        json.dump({str(l): layers[l] for l in L}, open(sys.argv[sys.argv.index('--json') + 1], 'w'))

if __name__ == '__main__':
    main()
