from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from huggingface_hub.utils.tqdm import disable_progress_bars
from model2vec import StaticModel
from vicinity.backends.basic import CosineBasicBackend
from vicinity.datatypes import QueryResult
from vicinity.utils import normalize

from semble.types import Chunk
from semble.utils import resolve_model_name

_EMBED_MIN_CHUNKS_FOR_PARALLEL = 2048
_EMBED_MAX_WORKERS = min(os.cpu_count() or 1, 8)


@cache
def _load_cached(model_path: str) -> StaticModel:
    """Load a model and cache it, but only after the path resolves."""
    # Disable HF progress bars since the model is loaded silently in the background during indexing.
    disable_progress_bars()
    try:
        model = StaticModel.from_pretrained(model_path, force_download=False)
    finally:
        disable_progress_bars()

    return model


def load_model(model_path: str | None = None) -> tuple[StaticModel, str]:
    """Return the current model, loading the default if none was provided."""
    if model_path is None:
        model_path = resolve_model_name()
    model = _load_cached(model_path)
    return model, model_path


def embed_chunks(model: StaticModel, chunks: list[Chunk]) -> npt.NDArray[np.float32]:
    """Embed chunks using the configured model.

    Splits work across threads when chunk count exceeds the parallel threshold.
    Both the HuggingFace tokenizer (Rust) and the numpy pooling ops release the
    GIL, so threads provide real parallelism on multi-core machines.
    """
    if not chunks:
        return np.empty((0, model.dim), dtype=np.float32)

    texts = [c.content for c in chunks]

    if len(chunks) < _EMBED_MIN_CHUNKS_FOR_PARALLEL or _EMBED_MAX_WORKERS <= 1:
        return np.array(model.encode(texts, use_multiprocessing=False), dtype=np.float32)

    batch_size = math.ceil(len(texts) / _EMBED_MAX_WORKERS)
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    def _encode(batch: list[str]) -> npt.NDArray[np.float32]:
        return np.array(model.encode(batch, use_multiprocessing=False), dtype=np.float32)

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        results = list(executor.map(_encode, batches))

    return np.concatenate(results, axis=0)


class SelectableBasicBackend(CosineBasicBackend):
    def _selector_dist(self, x: npt.NDArray, selector: npt.NDArray[np.int_]) -> npt.NDArray:
        """Compute cosine distance."""
        x_norm = normalize(x)
        sim = x_norm.dot(self._vectors[selector].T)
        return 1 - sim

    def query(self, vectors: npt.NDArray, k: int, selector: npt.NDArray[np.int_] | None = None) -> QueryResult:
        """Batched distance query.

        :param vectors: The vectors to query.
        :param k: The number of nearest neighbors to return.
        :param selector: Optional array of chunk indices to filter results by.
        :return: A list of tuples with the indices and distances.
        :raises ValueError: If k is less than 1.
        """
        if k < 1:
            raise ValueError(f"k should be >= 1, is now {k}")

        out: QueryResult = []
        num_vectors = len(self.vectors)
        effective_k = min(k, num_vectors)
        if selector is not None:
            effective_k = min(effective_k, len(selector))

        # Batch the queries
        for index in range(0, len(vectors), 1024):
            batch = vectors[index : index + 1024]
            if selector is not None:
                distances = self._selector_dist(batch, selector)
            else:
                distances = self._dist(batch)

            # Efficiently get the k smallest distances
            indices = np.argpartition(distances, kth=effective_k - 1, axis=1)[:, :effective_k]
            sorted_indices = np.take_along_axis(
                indices, np.argsort(np.take_along_axis(distances, indices, axis=1)), axis=1
            )
            sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)

            # Extend the output with tuples of (indices, distances)
            if selector is not None:
                sorted_indices = selector[sorted_indices]
            out.extend(zip(sorted_indices, sorted_distances))

        return out

    def save(self, path: Path) -> None:
        """Save the selectable basic backend."""
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)

    @classmethod
    def load(cls, path: Path) -> "SelectableBasicBackend":
        """Load a selectable basic backend.

        Materializes vectors into a contiguous float32 owned buffer and runs a
        one-shot dummy query so the first real search after a cache hit does not
        pay page-fault / OpenBLAS packing latency (can be 5–10× slower otherwise).
        """
        loaded = super().load(path)
        # Force an owned contiguous copy. ascontiguousarray is a no-op when the
        # loaded buffer is already C-contiguous, leaving a view that still pays
        # first-touch / page-fault cost on the first real query.
        vectors = np.array(loaded.vectors, dtype=np.float32, order="C", copy=True)
        backend = SelectableBasicBackend(vectors, loaded.arguments)
        if len(backend.vectors):
            backend.query(backend.vectors[:1].copy(), k=1)
        return backend
