"""Read-only TEM memory accounting; logical payload, not allocator/peak memory."""
import hashlib
import json
import platform
import sys
import uuid
from collections import Counter
from pathlib import Path


def tensor_summary(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": int(value.numel() * value.element_size())}


def replay_summary(vectors, labels):
    vectors = vectors if isinstance(vectors, list) else [vectors]
    labels = labels if isinstance(labels, list) else [labels]
    counts = Counter()
    for value in labels:
        counts.update(value.detach().cpu().reshape(-1).tolist())
    vector_parts = [tensor_summary(value) for value in vectors]
    label_parts = [tensor_summary(value) for value in labels]
    vector_bytes = sum(value["bytes"] for value in vector_parts)
    label_bytes = sum(value["bytes"] for value in label_parts)
    items = sum(counts.values())
    vector_items = sum(int(value.shape[0]) for value in vectors)
    if vector_items != items:
        raise ValueError("Replay vectors and labels have different item counts")
    return {"total_items": items,
            "items_per_class": {str(k): v for k, v in sorted(counts.items())},
            "vectors": vector_parts, "labels": label_parts,
            "vector_bytes": vector_bytes, "label_bytes": label_bytes,
            "tensor_bytes": vector_bytes + label_bytes,
            "MB": (vector_bytes + label_bytes) / 1_000_000,
            "MiB": (vector_bytes + label_bytes) / 1024**2}


def pending_summary(cache):
    if cache is None:
        return None
    graph = cache["graph"]
    # Do not call graph.edges(): DGL may materialize another sparse format.
    fields = {key: tensor_summary(cache[key])
              for key in ("topo_vecs", "labels_all")}
    index_count = sum(len(ids) for ids in cache["ids_per_cls_train"])
    return {"purpose": "pending_selection_training_state",
            "tensors": fields,
            "tensor_logical_bytes": sum(x["bytes"] for x in fields.values()),
            "training_index_count": index_count,
            "training_indices_int64_equivalent_bytes": index_count * 8,
            "graph_nodes": int(graph.num_nodes()), "graph_edges": int(graph.num_edges()),
            "graph_ndata": {str(k): tensor_summary(v) for k, v in graph.ndata.items()},
            "graph_edata": {str(k): tensor_summary(v) for k, v in graph.edata.items()},
            "graph_structure_actual_bytes": None,
            "note": "Logical payload may alias other state. Python index overhead and DGL "
                    "sparse-format storage are unmeasured; do not add these into a physical total."}


class MemoryAudit:
    def __init__(self, args, budget):
        self.enabled = bool(getattr(args, "memory_audit", False))
        if not self.enabled:
            return
        self.run_id = uuid.uuid4().hex
        self.path = Path(args.result_path) / "memory_audit" / (self.run_id + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.calls = Counter()
        self.context = {
            "run_id": self.run_id, "dataset": args.dataset,
            "method": args.method, "backbone": args.backbone, "ILmode": args.ILmode,
            "configuration_and_repeat": getattr(args, "current_model_save_path", None),
            "base_seed": args.seed,
            "seed_note": "repeat index is not an independently assigned seed",
            "budget_parameter": budget,
            "budget_definition": "per_class_cap" if isinstance(budget, int) else "per_class_ratio",
        }
        source = Path(__file__).resolve().parents[1]
        hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.rglob("*.py")}
        self.write("run_start", configuration=vars(args), source_sha256=hashes,
                   python=sys.version, platform=platform.platform(),
                   accounting="Logical tensor payload only; excludes peak memory, allocator, "
                              "evaluation caches and unmeasured graph/Python overhead")
        print("[memory_audit] " + str(self.path), flush=True)

    def write(self, event, **fields):
        if not self.enabled:
            return
        record = dict(self.context, event=event, **fields)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def snapshot(self, event, task, model, **fields):
        if self.enabled:
            self.write(event, task_index=int(task),
                       replay=replay_summary(model.TEM_vecs, model.TEM_labels),
                       pending_selection=pending_summary(model._prev_task_cache), **fields)

    def after_observe(self, args, task, model):
        if not self.enabled:
            return
        self.calls[int(task)] += 1
        if self.calls[int(task)] == args.epochs:
            self.snapshot("task_training_end", task, model,
                          observe_calls=self.calls[int(task)],
                          note="Current task remains pending; replay is updated on next task entry. "
                               "End detection assumes one full-batch observe call per epoch.")
