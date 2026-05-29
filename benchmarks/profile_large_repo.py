"""Profile semble on a large repository to identify performance bottlenecks.

Usage:
    uv run python -m benchmarks.profile_large_repo /path/to/large/repo
    uv run python -m benchmarks.profile_large_repo /path/to/large/repo --queries "auth middleware" "database connection"
    uv run python -m benchmarks.profile_large_repo /path/to/large/repo --cprofile

Measures:
    - Cold index build (full pipeline breakdown)
    - Cache validation time (warm run)
    - Query latency (p50/p90/p99)
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import numpy as np

# Patch before importing semble so we can time sub-steps
_timings: dict[str, list[float]] = {}


@contextmanager
def _timer(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    _timings.setdefault(label, []).append(elapsed)


def _instrument() -> None:
    """Monkey-patch hot paths at their call sites to collect fine-grained timings.

    Must patch the name in the *using* module, not the defining module, because
    each module captures a local reference at import time via `from x import f`.
    """
    # Ensure all modules are imported before we patch their namespaces.
    import semble.index.create as create_mod
    import semble.index.index as index_mod
    import bm25s

    # create_mod.walk_files — used in create_index_from_path
    _orig_walk = create_mod.walk_files
    def _timed_walk(root, extensions, ignore=None):
        with _timer("walk_files"):
            result = list(_orig_walk(root, extensions, ignore))
        return iter(result)
    create_mod.walk_files = _timed_walk  # type: ignore[assignment]

    # create_mod.embed_chunks — used in create_index_from_path
    _orig_embed = create_mod.embed_chunks
    def _timed_embed(model, chunks):
        with _timer("embed_chunks"):
            return _orig_embed(model, chunks)
    create_mod.embed_chunks = _timed_embed  # type: ignore[assignment]

    # create_mod.chunk_source — used in create_index_from_path
    _orig_chunk_source = create_mod.chunk_source
    def _timed_chunk_source(source, file_path, language):
        with _timer("chunk_source"):
            return _orig_chunk_source(source, file_path, language)
    create_mod.chunk_source = _timed_chunk_source  # type: ignore[assignment]

    # index_mod.get_validated_cache — used in SembleIndex.from_path
    _orig_validate = index_mod.get_validated_cache
    def _timed_validate(path, model_path, content):
        with _timer("get_validated_cache"):
            return _orig_validate(path, model_path, content)
    index_mod.get_validated_cache = _timed_validate  # type: ignore[assignment]

    # bm25s.BM25.index — method on the class, direct patch works
    _orig_bm25_index = bm25s.BM25.index
    def _timed_bm25_index(self, corpus, *args, **kwargs):
        with _timer("bm25_index"):
            return _orig_bm25_index(self, corpus, *args, **kwargs)
    bm25s.BM25.index = _timed_bm25_index  # type: ignore[method-assign]

    # create_mod.tokenize — the list comp arg is evaluated before bm25s.BM25.index
    # is called, so wrapping at the call site captures the true tokenization time.
    _orig_tokenize = create_mod.tokenize
    def _timed_tokenize(text):
        with _timer("tokenize"):
            return _orig_tokenize(text)
    create_mod.tokenize = _timed_tokenize  # type: ignore[assignment]


@dataclass
class PhaseResult:
    label: str
    ms: float


@dataclass
class ProfilingReport:
    repo_path: str
    file_count: int
    chunk_count: int
    cold_total_ms: float
    warm_total_ms: float
    cold_phases: list[PhaseResult] = field(default_factory=list)
    query_latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 50)) if self.query_latencies_ms else 0.0

    @property
    def p90_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 90)) if self.query_latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 99)) if self.query_latencies_ms else 0.0

    def print(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Profiling: {self.repo_path}")
        print(f"  Files indexed: {self.file_count:,}")
        print(f"  Chunks created: {self.chunk_count:,}")
        print(f"{'=' * 60}")

        print(f"\n[COLD BUILD]  total: {self.cold_total_ms:,.0f}ms")
        for phase in self.cold_phases:
            pct = (phase.ms / self.cold_total_ms * 100) if self.cold_total_ms else 0
            bar = "█" * int(pct / 2)
            print(f"  {phase.label:<28} {phase.ms:>8,.0f}ms  {pct:5.1f}%  {bar}")

        unaccounted = self.cold_total_ms - sum(p.ms for p in self.cold_phases)
        if unaccounted > 0:
            pct = unaccounted / self.cold_total_ms * 100
            print(f"  {'(other)':<28} {unaccounted:>8,.0f}ms  {pct:5.1f}%")

        print(f"\n[WARM RUN]    total: {self.warm_total_ms:,.0f}ms  (cache hit)")

        if self.query_latencies_ms:
            print(f"\n[SEARCH] n={len(self.query_latencies_ms)} queries")
            print(f"  p50: {self.p50_ms:.1f}ms")
            print(f"  p90: {self.p90_ms:.1f}ms")
            print(f"  p99: {self.p99_ms:.1f}ms")

        print()


def _sum_timing(label: str) -> float:
    return sum(_timings.get(label, [0.0]))


def run_profile(
    repo_path: Path,
    queries: list[str],
    top_k: int,
    use_cprofile: bool,
) -> None:
    _instrument()

    from semble import SembleIndex
    from semble.cache import clear_cache

    repo_str = str(repo_path)

    # --- Cold build ---
    print(f"Clearing cache for {repo_str}...", file=sys.stderr)
    clear_cache(repo_str)
    _timings.clear()

    print("Cold build (no cache)...", file=sys.stderr)

    if use_cprofile:
        pr = cProfile.Profile()
        pr.enable()

    cold_start = time.perf_counter()
    index = SembleIndex.from_path(repo_path)
    cold_ms = (time.perf_counter() - cold_start) * 1000

    if use_cprofile:
        pr.disable()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(30)
        print("\n[cProfile top 30 by cumulative time]")
        print(buf.getvalue())

    # Save index before cache to get warm timing
    from semble.cache import save_index_to_cache
    save_index_to_cache(index, repo_str)

    cold_phases = [
        PhaseResult("walk_files", _sum_timing("walk_files")),
        PhaseResult("chunk_source (total)", _sum_timing("chunk_source")),
        PhaseResult("embed_chunks", _sum_timing("embed_chunks")),
        PhaseResult("tokenize (BM25 prep)", _sum_timing("tokenize")),
        PhaseResult("bm25_index", _sum_timing("bm25_index")),
    ]

    chunk_count = len(index.chunks)
    file_count = len(index.stats.languages) and index.stats.indexed_files

    # --- Warm run (cache hit) ---
    _timings.clear()
    print("Warm run (from cache)...", file=sys.stderr)
    warm_start = time.perf_counter()
    index2 = SembleIndex.from_path(repo_path)
    warm_ms = (time.perf_counter() - warm_start) * 1000

    # --- Search queries ---
    query_latencies: list[float] = []
    if queries:
        print(f"Running {len(queries)} queries x 5 reps...", file=sys.stderr)
        for query in queries:
            for _ in range(5):
                t0 = time.perf_counter()
                index2.search(query, top_k=top_k)
                query_latencies.append((time.perf_counter() - t0) * 1000)
    else:
        default_queries = [
            "authentication middleware",
            "database connection pool",
            "error handling",
            "file read write",
            "config parsing",
        ]
        print(f"Running {len(default_queries)} default queries x 5 reps...", file=sys.stderr)
        for query in default_queries:
            for _ in range(5):
                t0 = time.perf_counter()
                index2.search(query, top_k=top_k)
                query_latencies.append((time.perf_counter() - t0) * 1000)

    report = ProfilingReport(
        repo_path=repo_str,
        file_count=file_count,
        chunk_count=chunk_count,
        cold_total_ms=cold_ms,
        warm_total_ms=warm_ms,
        cold_phases=cold_phases,
        query_latencies_ms=query_latencies,
    )
    report.print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="Path to the repository to profile")
    parser.add_argument("--queries", nargs="*", default=[], help="Search queries to benchmark")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results per query (default: 10)")
    parser.add_argument("--cprofile", action="store_true", help="Also dump cProfile top-30 for cold build")
    args = parser.parse_args()

    repo_path = args.repo.resolve()
    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    run_profile(
        repo_path=repo_path,
        queries=args.queries,
        top_k=args.top_k,
        use_cprofile=args.cprofile,
    )


if __name__ == "__main__":
    main()
