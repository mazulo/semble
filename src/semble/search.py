from collections.abc import Callable

import bm25s
import numpy as np
import numpy.typing as npt

from semble.index.dense import SelectableBasicBackend
from semble.index.sparse import selector_to_mask
from semble.ranking import apply_query_boost, boost_multi_chunk_files, diversify_results, rerank_topk, resolve_alpha
from semble.tokens import tokenize
from semble.types import Chunk, Encoder, SearchMode, SearchResult

Reranker = Callable[[dict[Chunk, float], int], list[tuple[Chunk, float]]]

_RRF_K = 60


def _rrf_scores(scores: dict[Chunk, float]) -> dict[Chunk, float]:
    """Convert raw scores to RRF scores 1/(k + rank); higher raw score → rank 1."""
    if not scores:
        return scores
    ranked = sorted(scores, key=lambda c: -scores[c])
    return {chunk: 1.0 / (_RRF_K + rank) for rank, chunk in enumerate(ranked, 1)}


def search_semantic(
    query: str,
    model: Encoder,
    semantic_index: SelectableBasicBackend,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
    diversity: float | None = None,
    chunk_index: dict[Chunk, int] | None = None,
) -> list[SearchResult]:
    """Run semantic search for a query."""
    fetch_k = top_k * 2 if diversity is not None and chunk_index is not None else top_k
    query_embedding = model.encode([query])
    indices, scores = semantic_index.query(query_embedding, k=fetch_k, selector=selector)[0]
    # Vicinity returns cosine distance; convert to similarity so higher = better.
    results = [
        SearchResult(chunk=chunks[index], score=1.0 - float(distance), source=SearchMode.SEMANTIC)
        for index, distance in zip(indices, scores)
    ]
    if diversity is not None and chunk_index is not None:
        return diversify_results(results, top_k, diversity, semantic_index._vectors, chunk_index)
    return results


def _sort_top_k(arr: npt.NDArray, top_k: int) -> npt.NDArray[np.int_]:
    """Get the top k indices of an array in sort order."""
    neg_arr = -arr
    if top_k >= len(arr):
        return np.argsort(neg_arr)
    partitioned = np.argpartition(neg_arr, kth=top_k)[:top_k]
    return partitioned[np.argsort(neg_arr[partitioned])]


def search_bm25(
    query: str,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    selector: npt.NDArray[np.int_] | None,
) -> list[SearchResult]:
    """Return chunks ranked by BM25 score, excluding zero-score results."""
    tokens = tokenize(query)
    if not tokens:
        return []
    mask = selector_to_mask(selector, len(chunks))
    scores: npt.NDArray[np.float32] = bm25_index.get_scores(tokens, weight_mask=mask)
    indices = _sort_top_k(scores, top_k)

    # Exclude chunks with zero score, no query tokens matched.
    return [
        SearchResult(chunk=chunks[i], score=float(scores[i]), source=SearchMode.BM25) for i in indices if scores[i] > 0
    ]


def search_hybrid(
    query: str,
    model: Encoder,
    semantic_index: SelectableBasicBackend,
    bm25_index: bm25s.BM25,
    chunks: list[Chunk],
    top_k: int,
    alpha: float | None = None,
    selector: npt.NDArray[np.int_] | None = None,
    reranker: Reranker | None = None,
    semantic_filter: Reranker | None = None,
    skip_file_boost: bool = False,
    diversity: float | None = None,
    chunk_index: dict[Chunk, int] | None = None,
) -> list[SearchResult]:
    """Hybrid search: alpha-weighted combination of semantic and BM25 scores.

    Both score sets are converted to RRF scores before combining, so alpha has
    a consistent meaning regardless of raw score magnitude.

    :param query: Search query string.
    :param model: Embedding model for semantic search.
    :param semantic_index: Pre-built semantic (vector) index.
    :param bm25_index: Pre-built BM25 index.
    :param chunks: All indexed chunks (parallel to BM25 index).
    :param top_k: Number of results to return.
    :param alpha: Weight for semantic score (1-alpha goes to BM25). None = auto-detect based on query type.
    :param selector: Optional array of chunk indices to filter results by.
    :param reranker: Optional callable replacing the default rerank_topk step.
    :param semantic_filter: Optional callable applied to semantic candidates before RRF fusion.
    :param skip_file_boost: If True, skip the ``boost_multi_chunk_files`` step.
    :param diversity: DPP diversity weight; when set, fetches 2× candidates and re-ranks with DPP.
    :param chunk_index: Chunk-to-index mapping required when diversity is set.
    :return: List of search results sorted by combined score descending.
    """
    alpha_weight = resolve_alpha(query, alpha)

    # Over-fetch candidates so the merged pool is large enough after union and re-ranking.
    candidate_count = top_k * 5

    # When a semantic filter is provided, over-fetch so DPP has material to work with.
    semantic_fetch_count = candidate_count * 2 if semantic_filter is not None else candidate_count
    semantic = search_semantic(query, model, semantic_index, chunks, semantic_fetch_count, selector)
    semantic_scores: dict[Chunk, float] = {result.chunk: result.score for result in semantic}

    if semantic_filter is not None:
        filtered = semantic_filter(semantic_scores, candidate_count)
        semantic_scores = {chunk: score for chunk, score in filtered}

    normalized_semantic = _rrf_scores(semantic_scores)
    bm25_results = search_bm25(query, bm25_index, chunks, candidate_count, selector)
    normalized_bm25 = _rrf_scores({r.chunk: r.score for r in bm25_results if r.score})

    # Sort by the file path and start line to
    # counteract randomness introduces by hashing.
    all_candidates = sorted(
        {*normalized_semantic, *normalized_bm25},
        key=lambda c: c.start_line,
    )
    combined_scores: dict[Chunk, float] = {
        chunk: alpha_weight * normalized_semantic.get(chunk, 0.0)
        + (1.0 - alpha_weight) * normalized_bm25.get(chunk, 0.0)
        for chunk in all_candidates
    }

    # Boost files with multiple relevant chunks.
    if not skip_file_boost:
        boost_multi_chunk_files(combined_scores)
    # Boost queries with specific identifiers in them.
    combined_scores = apply_query_boost(combined_scores, query, chunks)
    fetch_k = top_k * 2 if diversity is not None and chunk_index is not None else top_k
    if reranker is not None:
        ranked = reranker(combined_scores, fetch_k)
    else:
        ranked = rerank_topk(combined_scores, fetch_k, penalise_paths=alpha_weight < 1.0)
    results = [SearchResult(chunk=chunk, score=score, source=SearchMode.HYBRID) for chunk, score in ranked]
    if diversity is not None and chunk_index is not None:
        return diversify_results(results, top_k, diversity, semantic_index._vectors, chunk_index)
    return results
