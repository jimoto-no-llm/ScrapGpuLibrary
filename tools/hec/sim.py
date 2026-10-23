#!/usr/bin/env python3
"""hec_sim: offline simulator for the hot-expert-cache design (docs/plans/HOT_EXPERT_CACHE.md, unit H2 / U-sim).

Input : routing traces written by adapters/ggml/tools/hec_trace.cpp (PREFIX.bin[.gz] + PREFIX.json).
Output: per (trace, placement policy, device set) hit rate h (overall, per layer, per token) and the decode tg of the design's cost model (section 6.1),
        plus the bias statistics of the traces, the prefill statistics (distinct experts per ubatch, streaming estimate of pp) and a cross-domain table.

Only the standard library is used (no numpy).

Cost model (section 6.1, per token): T_tok = T_FIX + sum_l max( max_j G_lj , C_l )
   G_lj = fixed_j + n_lj * t_exp_j      (only when n_lj > 0)      C_l = CPU_FIX + n_cpu_l * T_CPU  (only when n_cpu_l > 0)
   Unlike the design's 6.3 calculation (expectation inside the max) the max is taken per token and layer on the actual selections, so the
   randomness of the router is included; the hit rate is the share of the 8 selected experts per layer that are resident on any GPU tier.

Policies (placement of experts onto tiers RX, WX, GT730, GT710, GT430 in this order of frequency; the rest stays on the CPU):
   static-oracle   frequency of the evaluated decode segment itself (upper bound, not realizable)
   static-prefill  frequency of the prefill of the same prompt (design 4.1; what a real run can know)
   static-cross    frequency of other data: for natural text the other domains, for random tokens the other llama-bench runs
   lru             exclusive LRU cascade RX -> WX -> ... -> CPU, warmed by the prefill of the same prompt, lookup on the state at token start
   hyst            frequency (EMA, half-life HALF tokens) + hysteresis (design 4.3: alpha, horizon, upload budget per token), start = static-prefill
"""
import argparse
import collections
import gzip
import json
import math
import os
import sys

# ---------------------------------------------------------------- parameters of the design (section 1.2, 5.2, 6.1)
EXPERT_MB = 2.654208
T_FIX = 18.0           # ms, fixed part per token (design 6.1)
T_CPU = 0.35           # ms per expert on the CPU (design); --cpu measured switches to the per-layer values of MB-H1
CPU_FIX = 0.05
T_CPU_L = None         # per-layer ms per expert on the CPU, set by --cpu measured
N_Q41 = 6              # layers 0..5 have ffn_down in Q4_1 in unsloth's Qwen3-30B-A3B-Q4_0.gguf (found by hec_bench)
SERIAL_HOST = None     # per-tier host-side GPU cost (ms), set by --serial-host
Q41_ON_GPU = True      # --q41 none: experts of layers 0..5 are never put on a GPU tier (slot arrays are Q4_0 only)
# name, fixed ms, GB/s, upload ms per expert
TIERS = [
    ("RX",    0.07, 76.0, 1.9),
    ("WX",    0.30, 26.0, 7.4),
    ("GT730", 0.06, 25.0, 6.6),
    ("GT710", 0.06, 10.0, 6.6),
    ("GT430", 0.30, 12.5, 9.0),     # fixed = L = 0.3 ms (design 5.4 / 6.3 configuration D)
]
T_EXP = [EXPERT_MB / bw for (_, _, bw, _) in TIERS]       # ms per expert (MB / (GB/s) = ms)
# experts per layer: conservative set of design 6.3 and the slot estimates of 1.2 (RX 1050 / 48, WX 740 / 48, GT730 335 / 48, GT710 730 / 48, GT430 335 / 48)
CAPS = {
    "cons":  [19, 13, 7, 14, 7],
    "est12": [22, 15, 7, 15, 7],
}
CONFIGS = {"A:RX+WX": 2, "B:+GT730": 3, "C:+GT710": 4, "D:+GT430": 5}      # number of leading tiers used
STREAM_GBPS = 1.5      # prefill streaming to the RX (design 7.2: 1.4 to 1.63 GB/s)


# ---------------------------------------------------------------- trace
class Trace:
    def __init__(self, prefix):
        self.prefix = prefix
        self.name = os.path.basename(prefix)
        self.meta = json.load(open(prefix + ".json"))
        p = prefix + ".bin"
        self.data = gzip.open(p + ".gz", "rb").read() if (not os.path.exists(p) and os.path.exists(p + ".gz")) else open(p, "rb").read()
        self.L = self.meta["n_layer"]
        self.K = self.meta["n_used"]
        self.NE = self.meta["n_expert"]
        self.segs = {s["name"]: s for s in self.meta["segments"]}
        assert len(self.data) == self.meta["n_tokens"] * self.L * self.K, "size mismatch"

    def ids(self, t, l):
        o = (t * self.L + l) * self.K
        return self.data[o:o + self.K]

    def seg_range(self, name):
        s = self.segs[name]
        return s["start"], s["start"] + s["n"]

    def counts(self, ranges, skip_unobserved=True):
        """counts[l][e] over token ranges [(a,b),...]; entries 255 (layer not observed for that token) are skipped"""
        c = [[0] * self.NE for _ in range(self.L)]
        for a, b in ranges:
            for t in range(a, b):
                for l in range(self.L):
                    for e in self.ids(t, l):
                        if e != 255:
                            c[l][e] += 1
        return c

    def ubatches_in(self, a, b):
        return [(s, n) for s, n, ph in self.meta["ubatches"] if a <= s < b]


def add_counts(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


# ---------------------------------------------------------------- placements
def static_layer(counts, caps, ntier):
    """per layer: the most frequent caps[0] experts to tier 0, the next caps[1] to tier 1, ... (design 6.3). returns place[l][e] = tier or -1"""
    place = []
    for l, row in enumerate(counts):
        order = sorted(range(len(row)), key=lambda e: (-row[e], e))
        pl = [-1] * len(row)
        if not Q41_ON_GPU and l < N_Q41:
            place.append(pl)
            continue
        i = 0
        for j in range(ntier):
            for e in order[i:i + caps[j]]:
                pl[e] = j
            i += caps[j]
        place.append(pl)
    return place


def static_global(counts, caps, ntier):
    """design 4.2: one greedy ranking over all (layer, expert); tier j gets caps[j] * n_layer slots in total"""
    L = len(counts)
    pairs = sorted(((l, e) for l in range(L) for e in range(len(counts[l])) if Q41_ON_GPU or l >= N_Q41), key=lambda p: (-counts[p[0]][p[1]], p[0], p[1]))
    place = [[-1] * len(counts[0]) for _ in range(L)]
    i = 0
    for j in range(ntier):
        for (l, e) in pairs[i:i + caps[j] * L]:
            place[l][e] = j
        i += caps[j] * L
    return place


# ---------------------------------------------------------------- cost
def tcpu(l):
    return T_CPU if T_CPU_L is None else T_CPU_L[l]


def layer_time(cnt, ncpu, l, ntier):
    """time of one layer. design model: devices and the CPU run in parallel, max(). --serial-host (MB-H2): the host submits to each used GPU
    and waits for it in the same thread; the host-side cost (command submit + sync + read back) OVH[j] of every used GPU is serial and only the GPU
    compute n * T_EXP is hidden behind the CPU part: layer = sum(OVH) + max(CPU, max GPU compute)"""
    c = (CPU_FIX + ncpu * tcpu(l)) if ncpu else 0.0
    if SERIAL_HOST is None:
        tm = c
        for j in range(ntier):
            if cnt[j]:
                g = TIERS[j][1] + cnt[j] * T_EXP[j]
                if g > tm:
                    tm = g
        return tm
    ovh = 0.0
    gm = 0.0
    for j in range(ntier):
        if cnt[j]:
            ovh += SERIAL_HOST[j]
            g = cnt[j] * T_EXP[j]
            if g > gm:
                gm = g
    return ovh + (c if c > gm else gm)


def token_cost(place, tr, t, ntier, per_layer=None):
    """returns (ms, hits_gpu, hits_by_tier[], per-layer hit counts) for token t, given a placement valid at the start of the token"""
    total = 0.0
    hits = 0
    by = [0] * ntier
    for l in range(tr.L):
        pl = place[l]
        cnt = [0] * ntier
        ncpu = 0
        for e in tr.ids(t, l):
            j = pl[e]
            if 0 <= j < ntier:
                cnt[j] += 1
            else:
                ncpu += 1
        tm = layer_time(cnt, ncpu, l, ntier)
        h = 0
        for j in range(ntier):
            if cnt[j]:
                by[j] += cnt[j]
                h += cnt[j]
        hits += h
        if per_layer is not None:
            per_layer[l] += h
        total += tm
    return T_FIX + total, hits, by


def tg_of(ms):
    return 1000.0 / ms


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def eval_static(tr, place, rng, ntier):
    ms_list, h_list = [], []
    pl = [0] * tr.L
    by_tot = [0] * ntier
    for t in range(*rng):
        ms, hits, by = token_cost(place, tr, t, ntier, pl)
        ms_list.append(ms)
        h_list.append(hits / (tr.L * tr.K))
        for j in range(ntier):
            by_tot[j] += by[j]
    n = len(ms_list)
    return summarize(ms_list, h_list, [x / (n * tr.K) for x in pl], [b / (n * tr.L * tr.K) for b in by_tot], n)


def summarize(ms_list, h_list, h_layer_counts, by_share, n, extra=None):
    mean_ms = sum(ms_list) / n
    d = {
        "n_tokens": n,
        "h": round(sum(h_list) / n, 4),
        "h_tok_p10": round(pct(h_list, 0.1), 4), "h_tok_p90": round(pct(h_list, 0.9), 4),
        "h_layer": [round(x, 3) for x in h_layer_counts],
        "h_tier": [round(x, 4) for x in by_share],
        "ms_per_token": round(mean_ms, 2),
        "tg": round(tg_of(mean_ms), 2),
        "tg_of_mean_tok": round(sum(tg_of(m) for m in ms_list) / n, 2),
    }
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------- dynamic policies
class LRUCascade:
    def __init__(self, caps_total, ntier):
        self.cap = caps_total[:ntier]
        self.n = ntier
        self.tiers = [collections.OrderedDict() for _ in range(ntier)]
        self.where = {}
        self.uploads = [0] * ntier

    def place_of(self, l, e):
        return self.where.get((l, e), -1)

    def _insert(self, j, key):
        # insert at MRU of tier j; cascade the LRU victim to tier j+1 (an upload there), drop it after the last tier
        while True:
            if j >= self.n:
                return
            T = self.tiers[j]
            T[key] = True
            self.where[key] = j
            self.uploads[j] += 1
            if len(T) <= self.cap[j]:
                return
            victim, _ = T.popitem(last=False)
            del self.where[victim]
            key, j = victim, j + 1

    def access(self, key):
        j = self.where.get(key, -1)
        if j == 0:
            self.tiers[0].move_to_end(key)
            return
        if j > 0:
            del self.tiers[j][key]
            del self.where[key]
        self._insert(0, key)


def eval_lru(tr, warm_ranges, rng, ntier, caps):
    caps_total = [c * tr.L for c in caps]
    lru = LRUCascade(caps_total, ntier)
    for a, b in warm_ranges:
        for t in range(a, b):
            for l in range(tr.L):
                for e in tr.ids(t, l):
                    if e != 255:
                        lru.access((l, int(e)))
    lru.uploads = [0] * ntier
    ms_list, h_list = [], []
    pl = [0] * tr.L
    by_tot = [0] * ntier
    for t in range(*rng):
        # lookups on the state at the start of the token (an expert uploaded during this token is not usable in this token)
        total = 0.0
        hits = 0
        for l in range(tr.L):
            cnt = [0] * ntier
            ncpu = 0
            for e in tr.ids(t, l):
                j = lru.where.get((l, e), -1)
                if j >= 0:
                    cnt[j] += 1
                else:
                    ncpu += 1
            tm = layer_time(cnt, ncpu, l, ntier)
            for j in range(ntier):
                if cnt[j]:
                    by_tot[j] += cnt[j]
                    pl[l] += cnt[j]
                    hits += cnt[j]
            total += tm
        ms_list.append(T_FIX + total)
        h_list.append(hits / (tr.L * tr.K))
        for l in range(tr.L):
            for e in tr.ids(t, l):
                lru.access((l, int(e)))
    n = len(ms_list)
    up_tok = [u / n for u in lru.uploads]
    up_ms = sum(up_tok[j] * TIERS[j][3] for j in range(ntier))
    ex = {"uploads_per_token": [round(x, 2) for x in up_tok], "upload_ms_per_token_if_serial": round(up_ms, 1),
          "tg_if_uploads_serial": round(tg_of(sum(ms_list) / n + up_ms), 2)}
    return summarize(ms_list, h_list, [x / n / tr.K for x in pl], [b / (n * tr.L * tr.K) for b in by_tot], n, ex)


def eval_hyst(tr, init_counts, prefill_ranges, rng, ntier, caps, half=256.0, alpha=0.25, horizon=256, budget=None):
    """frequency + hysteresis (design 4.3). EMA counters per (layer, expert) in units of selections per token, warmed on the prefill tokens.
    A CPU expert e replaces the weakest GPU expert v of the same layer when f_e > (1+alpha) f_v and (f_e - f_v) * dt * horizon > upload (ms),
    dt = T_CPU - T_EXP[tier of v]; at most budget[j] experts per token and tier (accumulated as credit; default RX 1, others 0.25)."""
    if budget is None:
        budget = [1.0] + [0.25] * (ntier - 1)
    place = static_layer(init_counts, caps, ntier)
    d = 0.5 ** (1.0 / half)
    f = [[0.0] * tr.NE for _ in range(tr.L)]
    # warm start: EMA over the prefill tokens (selection indicator per token)
    for a, b in prefill_ranges:
        for t in range(a, b):
            for l in range(tr.L):
                row = f[l]
                for e in range(tr.NE):
                    row[e] *= d
                for e in tr.ids(t, l):
                    if e != 255:
                        row[e] += 1.0 - d
    credit = [0.0] * ntier
    swaps = [0] * ntier
    ms_list, h_list = [], []
    pl = [0] * tr.L
    by_tot = [0] * ntier
    for t in range(*rng):
        ms, hits, by = token_cost(place, tr, t, ntier, pl)
        ms_list.append(ms)
        h_list.append(hits / (tr.L * tr.K))
        for j in range(ntier):
            by_tot[j] += by[j]
        for j in range(ntier):
            credit[j] = min(credit[j] + budget[j], 2.0)
        for l in range(tr.L):
            row = f[l]
            for e in range(tr.NE):
                row[e] *= d
            for e in tr.ids(t, l):
                row[e] += 1.0 - d
            # candidate: strongest CPU expert; victim: weakest GPU expert of this layer (swap target = victim's tier)
            pl_l = place[l]
            ce, cf = -1, -1.0
            ve, vf = -1, 1e9
            for e in range(tr.NE):
                if pl_l[e] < 0:
                    if row[e] > cf:
                        cf, ce = row[e], e
                elif row[e] < vf:
                    vf, ve = row[e], e
            if ce < 0 or ve < 0:
                continue
            j = pl_l[ve]
            if cf > (1 + alpha) * vf and (cf - vf) * (tcpu(l) - T_EXP[j]) * horizon > TIERS[j][3] and credit[j] >= 1.0:
                credit[j] -= 1.0
                pl_l[ve] = -1
                pl_l[ce] = j
                swaps[j] += 1
    n = len(ms_list)
    sw_tok = [s / n for s in swaps]
    up_ms = sum(sw_tok[j] * TIERS[j][3] for j in range(ntier))
    ex = {"swaps_per_token": [round(x, 3) for x in sw_tok], "upload_ms_per_token_if_serial": round(up_ms, 1),
          "tg_if_uploads_serial": round(tg_of(sum(ms_list) / n + up_ms), 2)}
    return summarize(ms_list, h_list, [x / n / tr.K for x in pl], [b / (n * tr.L * tr.K) for b in by_tot], n, ex)


# ---------------------------------------------------------------- bias statistics
def bias_stats(counts_layer_rows, L_obs=None):
    """for per-layer count rows: normalised entropy H/Hmax, #experts covering 80 % of selections, top-32 share (self counts)"""
    ents, c80, top32 = [], [], []
    for row in counts_layer_rows:
        tot = sum(row)
        if tot == 0:
            continue
        p = [x / tot for x in row if x]
        ents.append(-sum(x * math.log(x) for x in p) / math.log(len(row)))
        s = sorted(row, reverse=True)
        acc = 0
        for i, x in enumerate(s, 1):
            acc += x
            if acc >= 0.8 * tot:
                c80.append(i)
                break
        top32.append(sum(s[:32]) / tot)
    m = lambda v: sum(v) / len(v) if v else float("nan")
    return {"H/Hmax": round(m(ents), 3), "experts_for_80pct": round(m(c80), 1), "top32_share_self": round(m(top32), 3)}


def coverage(src_counts, dst_counts, k=32):
    """share of dst selections that fall into the src top-k experts of the same layer (held-out predictive power of src for dst)"""
    cov = []
    for rs, rd in zip(src_counts, dst_counts):
        tot = sum(rd)
        if tot == 0:
            continue
        order = sorted(range(len(rs)), key=lambda e: (-rs[e], e))[:k]
        cov.append(sum(rd[e] for e in order) / tot)
    return sum(cov) / len(cov)


# ---------------------------------------------------------------- prefill statistics
def prefill_stats(tr, seg_names, place_by_cfg):
    """per ubatch and layer: distinct experts used, tokens per used expert; streaming estimate of pp for each placement"""
    res = {"distinct_per_layer_ubatch": None, "share_experts_with_le2_tokens": None, "stream_pp": {}}
    dist_tot = n_ub = 0
    le2 = used = 0
    stream = {k: 0.0 for k in place_by_cfg}
    ntok = 0
    for name in seg_names:
        a, b = tr.seg_range(name)
        for s, n in tr.ubatches_in(a, b):
            if n < 2:
                continue
            ntok += n
            n_ub += 1
            for l in range(tr.L - 1):                       # last layer: only the logits row is observed in prefill
                c = collections.Counter()
                for t in range(s, s + n):
                    for e in tr.ids(t, l):
                        c[e] += 1
                dist_tot += len(c)
                used += len(c)
                le2 += sum(1 for v in c.values() if v <= 2)
                for k, pl in place_by_cfg.items():
                    miss = sum(1 for e in c if pl[l][e] < 0)
                    stream[k] += miss * EXPERT_MB / (STREAM_GBPS * 1000.0) * 1000.0     # ms
    if n_ub:
        res["distinct_per_layer_ubatch"] = round(dist_tot / (n_ub * (tr.L - 1)), 1)
        res["share_experts_with_le2_tokens"] = round(le2 / used, 3)
        for k in stream:
            sec = stream[k] / 1000.0                               # layer 47 sees one token (inp_out_ids) in prefill: negligible, not scaled
            res["stream_pp"][k] = round(ntok / sec, 1) if sec else None
    return res



# ---------------------------------------------------------------- reproduction of the design's 6.3 table (expectation model on imatrix counts)
def expected_tg(place, dst_counts, ntier):
    """design 6.3: expected number of selections per tier from the destination frequencies, max over devices and the CPU per layer, summed"""
    total = T_FIX
    hsum = 0.0
    for l, row in enumerate(dst_counts):
        tot = float(sum(row))
        n = [0.0] * ntier
        ncpu = 0.0
        for e, c in enumerate(row):
            j = place[l][e]
            if 0 <= j < ntier:
                n[j] += 8 * c / tot
            else:
                ncpu += 8 * c / tot
        tm = (CPU_FIX + ncpu * tcpu(l)) if ncpu > 0 else 0.0
        for j in range(ntier):
            if n[j] > 0:
                tm = max(tm, TIERS[j][1] + n[j] * T_EXP[j])
        total += tm
        hsum += (8 - ncpu) / 8
    return hsum / len(dst_counts), 1000.0 / total


def reproduce(imat_dir):
    ld = lambda n: [json.load(open(os.path.join(imat_dir, n + ".counts.json")))[str(l)] for l in range(48)]
    cases = [("readme (h1->h2)", "readme_h1", "readme_h2"), ("py (h1->h2)", "py_code_h1", "py_code_h2"),
             ("readme->en_legal", "readme", "en_legal"), ("readme->py", "readme", "py_code")]
    doc = {"A:RX+WX": [(0.76, 18.9), (0.83, 22.6), (0.26, 8.3), (0.38, 9.7)], "B:+GT730": [(0.82, 21.8), (0.86, 24.5), (0.30, 8.7), (0.44, 10.4)],
           "C:+GT710": [(0.90, 25.3), (0.92, 26.2), (0.41, 10.0), (0.57, 12.7)], "D:+GT430": [(0.93, 25.7), (0.94, 26.4), (0.46, 10.7), (0.62, 14.0)]}
    rows = []
    for cname, nt in CONFIGS.items():
        cells = []
        for i, (lab, src, dst) in enumerate(cases):
            h, tg = expected_tg(static_layer(ld(src), CAPS["cons"], nt), ld(dst), nt)
            cells.append(f"{h:.2f} / {tg:.1f} (doc {doc[cname][i][0]:.2f} / {doc[cname][i][1]:.1f})")
        rows.append((cname, cells))
    print("| config | " + " | ".join(c[0] for c in cases) + " |")
    print("| --- | " + " | ".join("---" for _ in cases) + " |")
    for cname, cells in rows:
        print(f"| {cname} | " + " | ".join(cells) + " |")

# ---------------------------------------------------------------- driver
def fmt(r, key="tg"):
    return f"{r['h']:.3f} / {r[key]:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reproduce", metavar="IMAT_DIR", help="reproduce the design's 6.3 table from the imatrix *.counts.json in IMAT_DIR and exit")
    ap.add_argument("--trace", action="append", help="trace prefix (PREFIX.json + PREFIX.bin[.gz]); the bench trace has segments pp*/tg*/pg*")
    ap.add_argument("--out", default="hec-sim.json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--caps", default="cons", choices=list(CAPS))
    ap.add_argument("--half", type=float, default=256.0)
    ap.add_argument("--cpu", default="design", choices=["design", "measured"], help="CPU time per expert: 0.35 ms (design) or MB-H1 per layer (0.276 ms Q4_0, 0.602 ms for the Q4_1-down layers 0-5, +0.027 ms)")
    ap.add_argument("--serial-host", default=None, metavar="RX[,WX,GT730,GT710,GT430]", help="MB-H2 model: per-layer host-side cost in ms of every used GPU (submit + sync + read back), serial in the host thread; missing values default to the design's fixed cost")
    ap.add_argument("--q41", default="gpu", choices=["gpu", "none"], help="none: layers 0-5 (ffn_down Q4_1) experts stay on the CPU (slot arrays are Q4_0)")
    ap.add_argument("--quick", action="store_true", help="skip the dynamic policies")
    a = ap.parse_args()
    if a.reproduce:
        return reproduce(a.reproduce)
    if not a.trace:
        ap.error('--trace is required')
    global T_CPU_L, CPU_FIX, Q41_ON_GPU, SERIAL_HOST
    caps = CAPS[a.caps]
    if a.cpu == "measured":
        T_CPU_L = [0.602 if l < N_Q41 else 0.276 for l in range(48)]
        CPU_FIX = 0.027
    Q41_ON_GPU = a.q41 == "gpu"
    if a.serial_host:
        v = [float(x) for x in a.serial_host.split(",")]
        SERIAL_HOST = [v[j] if j < len(v) else TIERS[j][1] for j in range(len(TIERS))]
    traces = [Trace(p) for p in a.trace]
    bench = [t for t in traces if any(n.startswith("tg") for n in t.segs)]
    natural = [t for t in traces if t not in bench]

    # scenarios: label, trace, decode segment, own prefill segment (or None), segments used as "other data" for static-cross
    scen = []
    for tr in bench:
        names = sorted(tr.segs)
        for n in names:
            if n.endswith("_tg") or (n.startswith("tg") and "_pp" not in n):
                own = n[:-3] + "_pp" if n.endswith("_tg") else None
                scen.append({"label": "random:" + n, "tr": tr, "dec": n, "own_pf": own,
                             "cross": [(tr, m) for m in names if m != n and m != own]})
    for tr in natural:
        base = [n for n in tr.segs if n.endswith("_decode")][0][:-7]
        cross = []
        for o in natural:
            if o is tr:
                continue
            ob = [n for n in o.segs if n.endswith("_decode")][0][:-7]
            cross += [(o, ob + "_prefill"), (o, ob + "_decode")]
        scen.append({"label": "text:" + (base[:-4] if base.endswith(".txt") else base), "tr": tr, "dec": base + "_decode", "own_pf": base + "_prefill", "cross": cross})

    out = {"params": {"caps": caps, "tiers": [t[0] for t in TIERS], "T_FIX": T_FIX, "T_CPU": T_CPU, "half": a.half, "cpu": a.cpu, "q41": a.q41, "serial_host": SERIAL_HOST}, "scenarios": {}, "bias": {}, "prefill": {}, "cross_domain": {}}
    md = []
    log = lambda s: (print(s, file=sys.stderr), None)[1]

    # ---- bias statistics + held-out predictive power
    for s in scen:
        tr = s["tr"]
        dec = tr.counts([tr.seg_range(s["dec"])])
        b = bias_stats(dec)
        entry = {"decode_self": b}
        if s["own_pf"]:
            pf = tr.counts([tr.seg_range(s["own_pf"])])
            entry["prefill_self"] = bias_stats(pf)
            entry["prefill_top32_covers_decode"] = round(coverage(pf, dec), 3)
        cross_c = None
        for (o, seg) in s["cross"]:
            c = o.counts([o.seg_range(seg)])
            cross_c = c if cross_c is None else add_counts(cross_c, c)
        if cross_c is not None:
            entry["cross_top32_covers_decode"] = round(coverage(cross_c, dec), 3)
        s["_dec_counts"] = dec
        s["_cross_counts"] = cross_c
        out["bias"][s["label"]] = entry
        log(f"bias {s['label']}: {entry}")

    # ---- policies
    for s in scen:
        tr = s["tr"]
        rng = tr.seg_range(s["dec"])
        pf_rng = [tr.seg_range(s["own_pf"])] if s["own_pf"] else []
        sc = {}
        sources = {"static-oracle": s["_dec_counts"]}
        if s["_cross_counts"] is not None:
            sources["static-cross"] = s["_cross_counts"]
        if s["own_pf"]:
            sources["static-prefill"] = tr.counts(pf_rng)
        for pol, cnt in sources.items():
            for cname, nt in CONFIGS.items():
                sc[f"{pol}|{cname}"] = eval_static(tr, static_layer(cnt, caps, nt), rng, nt)
            sc[f"{pol}-global|C:+GT710"] = eval_static(tr, static_global(cnt, caps, 4), rng, 4)
        if not a.quick:
            warm = pf_rng
            init = sources.get("static-prefill", sources.get("static-cross", sources["static-oracle"]))
            for cname in ("B:+GT730", "C:+GT710"):
                nt = CONFIGS[cname]
                sc[f"lru|{cname}"] = eval_lru(tr, warm, rng, nt, caps)
                sc[f"hyst|{cname}"] = eval_hyst(tr, init, warm, rng, nt, caps, half=a.half)
        out["scenarios"][s["label"]] = sc
        log(f"policies {s['label']}: " + "; ".join(f"{k}={v['h']:.2f}/{v['tg']:.1f}" for k, v in sc.items() if k.endswith("C:+GT710")))

    # ---- cross-domain static (config C): placement from domain X prefill+decode, evaluated on domain Y decode
    for s in scen:
        if not s["label"].startswith("text:"):
            continue
        row = {}
        for o in scen:
            if not o["label"].startswith("text:"):
                continue
            cnt = o["tr"].counts([o["tr"].seg_range(o["own_pf"]), o["tr"].seg_range(o["dec"])]) if o is not s else None
            if o is s:
                cnt = o["tr"].counts([o["tr"].seg_range(o["own_pf"])])      # diagonal = placement from the own prefill
            r = eval_static(s["tr"], static_layer(cnt, caps, 4), s["tr"].seg_range(s["dec"]), 4)
            row[o["label"]] = {"h": r["h"], "tg": r["tg"]}
        out["cross_domain"][s["label"]] = row

    # ---- prefill
    for s in scen:
        tr = s["tr"]
        if not s["own_pf"]:
            continue
        place = {}
        for cname in ("A:RX+WX", "C:+GT710", "D:+GT430"):
            nt = CONFIGS[cname]
            cnt = s["_cross_counts"] if s["_cross_counts"] is not None else s["tr"].counts([tr.seg_range(s["own_pf"])])
            place[cname + (" (placement from other data)" if s["_cross_counts"] is not None else " (placement from own prefill)")] = static_layer(cnt, caps, nt)
        place["none"] = [[-1] * tr.NE for _ in range(tr.L)]
        out["prefill"][s["label"]] = prefill_stats(tr, [s["own_pf"]], place)

    json.dump(out, open(a.out, "w"), indent=1)
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
