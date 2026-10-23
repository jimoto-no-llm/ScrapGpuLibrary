# Large logs

Log files over 200 KB under `docs/benchmarks/moe-ep-20261021/`, `moe-ep-20261022/` and `moe-ep-20261023/` were gzipped on 2026-10-24 to keep the tree small. A README that names `logs/foo.log` may now mean `logs/foo.log.gz` (`zcat` / `zless`).
