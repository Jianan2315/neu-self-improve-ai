"""Wikipedia-2018 / E5 retrieval for the Week 08 assignment.

Resources and ordering follow the official Search-R1 index/corpus release.
Downloads and model loading occur only in explicit calls, never on import.
"""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
from collections import OrderedDict
from copy import deepcopy


def prepare_assets(directory):
    from huggingface_hub import HfApi, hf_hub_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for filename in ("e5_Flat.index", "wiki-18.jsonl"):
            if not (directory / filename).is_file():
                raise FileNotFoundError(f"Incomplete retrieval assets: {filename}")
        return manifest
    api = HfApi()
    index_repo = "PeterJinGo/wiki-18-e5-index"
    corpus_repo = "PeterJinGo/wiki-18-corpus"
    index_revision = api.dataset_info(index_repo).sha
    corpus_revision = api.dataset_info(corpus_repo).sha
    encoder_id = "intfloat/e5-base-v2"
    encoder_revision = api.model_info(encoder_id).sha
    parts = []
    for name in ("part_aa", "part_ab"):
        print(f"Downloading {index_repo}/{name}", flush=True)
        parts.append(hf_hub_download(index_repo, name, repo_type="dataset", revision=index_revision,
                                     local_dir=directory / "downloads" / "index"))
    index_hash = hashlib.sha256()
    temporary = directory / "e5_Flat.index.partial"
    with temporary.open("wb") as output:
        for part in parts:
            with open(part, "rb") as source:
                for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    index_hash.update(block)
                    output.write(block)
    temporary.replace(directory / "e5_Flat.index")
    compressed = hf_hub_download(corpus_repo, "wiki-18.jsonl.gz", repo_type="dataset", revision=corpus_revision,
                                local_dir=directory / "downloads" / "corpus")
    corpus_hash = hashlib.sha256()
    temporary = directory / "wiki-18.jsonl.partial"
    with gzip.open(compressed, "rb") as source, temporary.open("wb") as output:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            corpus_hash.update(block)
            output.write(block)
    temporary.replace(directory / "wiki-18.jsonl")
    manifest = {"index_repo": index_repo, "index_revision": index_revision,
                "corpus_repo": corpus_repo, "corpus_revision": corpus_revision,
                "index_sha256": index_hash.hexdigest(), "corpus_sha256": corpus_hash.hexdigest(),
                "encoder_id": encoder_id, "encoder_revision": encoder_revision,
                "pooling": "attention-mask mean; L2 normalized", "query_prefix": "query: ",
                "query_max_length": 256, "backend": "faiss-cpu exact flat index"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class DenseRetriever:
    QUERY_CACHE_MAX_ENTRIES = 2048

    def clear_query_cache(self):
        """Cache lifetime is one loaded, immutable encoder/index/corpus instance."""
        self._query_cache = OrderedDict()

    def __init__(self, directory):
        import faiss
        import torch
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        faiss.omp_set_num_threads(16)
        torch.set_num_threads(16)
        print("Loading the full Wikipedia index into CPU RAM.", flush=True)
        self.index = faiss.read_index(str(self.directory / "e5_Flat.index"))
        self.corpus = load_dataset("json", data_files=str(self.directory / "wiki-18.jsonl"), split="train")
        if self.index.ntotal != len(self.corpus):
            raise ValueError("Index/corpus row counts differ; document ordering cannot be trusted.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.manifest["encoder_id"], revision=self.manifest["encoder_revision"])
        self.encoder = AutoModel.from_pretrained(self.manifest["encoder_id"], revision=self.manifest["encoder_revision"])
        self.encoder.eval()
        self.clear_query_cache()
        print(f"Retriever ready: {len(self.corpus)} passages.", flush=True)

    def retrieve(self, query, topk=3):
        return self.retrieve_many([query], topk)[0]

    def retrieve_many(self, queries, topk=3, batch_size=32, *, use_cache=True):
        """Batch exact searches, preserving order and duplicate-query results.

        Chunking bounds encoder memory. The corpus, index, pooling and top-k
        are unchanged; batching can introduce small floating-point differences.
        """
        import torch
        import time
        from search_r1_framework.retrieval_protocol import validate_request
        validate_request(queries, topk)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be a boolean.")
        if not hasattr(self, "_query_cache"):
            self.clear_query_cache()
        started = time.monotonic()
        unique = list(dict.fromkeys(queries))
        found = {}
        missing = []
        for query in unique:
            key = (query, topk)
            if use_cache and key in self._query_cache:
                self._query_cache.move_to_end(key)
                found[query] = self._query_cache[key]
            else:
                missing.append(query)
        cached = set(found)
        timings = {"encoding_seconds": 0.0, "index_search_seconds": 0.0,
                   "passage_read_seconds": 0.0, "query_count": len(queries),
                   "unique_query_count": len(unique), "batch_size": batch_size,
                   "cache_enabled": use_cache,
                   "cache_hit_queries": sum(q in cached for q in queries),
                   "cache_hit_unique_queries": len(cached),
                   "searched_unique_queries": len(missing)}
        for offset in range(0, len(missing), batch_size):
            batch = missing[offset:offset + batch_size]
            tick = time.monotonic()
            tokens = self.tokenizer(["query: " + q for q in batch], return_tensors="pt",
                                    padding=True, truncation=True, max_length=256)
            with torch.inference_mode():
                hidden = self.encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].bool().unsqueeze(-1)
                pooled = hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1)
                embedding = torch.nn.functional.normalize(pooled, dim=-1).numpy().astype("float32")
            if embedding.shape[1] != self.index.d:
                raise ValueError("Encoder dimension differs from the index.")
            timings["encoding_seconds"] += time.monotonic() - tick
            tick = time.monotonic()
            scores, indices = self.index.search(embedding, topk)
            timings["index_search_seconds"] += time.monotonic() - tick
            tick = time.monotonic()
            for query, score_row, index_row in zip(batch, scores, indices):
                found[query] = [{"id": str(int(i)), "contents": self.corpus[int(i)]["contents"],
                                 "score": float(score)}
                                for score, i in zip(score_row, index_row) if i >= 0]
            timings["passage_read_seconds"] += time.monotonic() - tick
        # Commit only successful searches; never cache exceptions or partial calls.
        if use_cache:
            for query in missing:
                key = (query, topk)
                self._query_cache[key] = found[query]
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self.QUERY_CACHE_MAX_ENTRIES:
                    self._query_cache.popitem(last=False)
        timings["cache_entries"] = len(self._query_cache)
        timings["backend_total_seconds"] = time.monotonic() - started
        self.last_batch_metrics = timings
        # Callers must not mutate a cached document through a returned reference.
        return [deepcopy(found[q]) for q in queries]
