"""
KG Relation Pattern Statistics (PyTorch-based)

Computes counts of relations exhibiting:
- symmetry:       r(h,t) implies r(t,h) for many pairs
- anti-symmetry:  r(h,t) and r(t,h) rarely occur except when h==t
- composition:    exists r1, r2 such that r1∘r2 approximates r (support/ratio)

Input formats supported (whitespace-separated triples):
- Folder with any of: train.txt, valid.txt, test.txt (FB15k-237 style)
- One or more files passed via --files

Usage examples:
  python kg_pattern_stats.py --data path/to/fb15k237
  python kg_pattern_stats.py --files path/to/train.txt path/to/valid.txt

Notes
- Uses PyTorch for vectorized symmetry/anti-symmetry checks.
- Composition is estimated via set-based joins for scalability.
- Thresholds are configurable; defaults are conservative.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch


Triple = Tuple[str, str, str]


def read_triples_from_file(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
    return triples


def read_triples(data_path: str | None, files: Sequence[str] | None) -> List[Triple]:
    paths: List[str] = []
    if files:
        paths.extend(list(files))
    if data_path:
        for name in ("train.txt", "valid.txt", "test.txt"):
            p = os.path.join(data_path, name)
            if os.path.exists(p):
                paths.append(p)
    if not paths:
        raise FileNotFoundError(
            "No triple files found. Pass --data <dir> or --files <paths>"
        )

    triples: List[Triple] = []
    for p in paths:
        triples.extend(read_triples_from_file(p))
    return triples


def read_triples_from_pykeen(dataset_spec: str) -> List[Triple]:
    """Load triples from a PyKEEN dataset specification.

    Accepts names like 'biokg', class names like 'BioKG', or fully-qualified
    'pykeen.datasets.biokg:BioKG'. Requires pykeen to be installed.
    """
    try:
        from pykeen.datasets import get_dataset
    except Exception as e:
        raise RuntimeError(
            "pykeen is required for --pykeen usage. Install with `pip install pykeen`."
        ) from e

    ds = None
    # Try direct get by spec
    try:
        ds = get_dataset(dataset=dataset_spec)
    except Exception:
        pass

    if ds is None:
        # Try class in pykeen.datasets namespace
        try:
            import importlib
            pk_ds = importlib.import_module("pykeen.datasets")
            if hasattr(pk_ds, dataset_spec):
                ds_cls = getattr(pk_ds, dataset_spec)
                ds = get_dataset(dataset=ds_cls)
        except Exception:
            ds = None

    if ds is None and ":" in dataset_spec:
        # Fully qualified path module:Class
        try:
            import importlib
            mod_name, cls_name = dataset_spec.split(":", 1)
            mod = importlib.import_module(mod_name)
            ds_cls = getattr(mod, cls_name)
            ds = get_dataset(dataset=ds_cls)
        except Exception:
            ds = None

    if ds is None:
        # last attempt: lowercased name
        try:
            ds = get_dataset(dataset=dataset_spec.lower())
        except Exception as e:
            raise RuntimeError(
                f"Could not resolve PyKEEN dataset spec: {dataset_spec}"
            ) from e

    def tf_to_label_triples(tf) -> List[Triple]:
        if tf is None:
            return []
        mt = tf.mapped_triples
        # Ensure on CPU and Python ints
        mt = mt.detach().cpu().tolist()
        # Build id->label maps
        e_map = getattr(tf, "entity_id_to_label", None)
        r_map = getattr(tf, "relation_id_to_label", None)
        if not e_map:
            # invert entity_to_id
            e2i = getattr(tf, "entity_to_id", {})
            e_map = {i: e for e, i in e2i.items()}
        if not r_map:
            r2i = getattr(tf, "relation_to_id", {})
            r_map = {i: r for r, i in r2i.items()}
        triples: List[Triple] = []
        for h, r, t in mt:
            h_l = e_map.get(int(h), str(h))
            r_l = r_map.get(int(r), str(r))
            t_l = e_map.get(int(t), str(t))
            triples.append((h_l, r_l, t_l))
        return triples

    triples: List[Triple] = []
    for split_name in ("training", "validation", "testing"):
        tf = getattr(ds, split_name, None)
        triples.extend(tf_to_label_triples(tf))
    return triples


def build_ids(triples: Iterable[Triple]):
    ent_to_id: Dict[str, int] = {}
    rel_to_id: Dict[str, int] = {}
    ents: List[str] = []
    rels: List[str] = []

    def get_ent(e: str) -> int:
        if e in ent_to_id:
            return ent_to_id[e]
        i = len(ents)
        ent_to_id[e] = i
        ents.append(e)
        return i

    def get_rel(r: str) -> int:
        if r in rel_to_id:
            return rel_to_id[r]
        i = len(rels)
        rel_to_id[r] = i
        rels.append(r)
        return i

    id_triples: List[Tuple[int, int, int]] = []
    for h, r, t in triples:
        id_triples.append((get_ent(h), get_rel(r), get_ent(t)))
    return id_triples, ents, rels, ent_to_id, rel_to_id


def per_relation_edges(
    id_triples: Sequence[Tuple[int, int, int]], n_ents: int
):
    rel2_edges: Dict[int, List[Tuple[int, int]]]= defaultdict(list)
    for h, r, t in id_triples:
        rel2_edges[r].append((h, t))

    # For composition we need adjacency maps
    rel2_head_to_tails: Dict[int, Dict[int, Set[int]]] = {}
    rel2_tail_to_heads: Dict[int, Dict[int, Set[int]]] = {}

    for r, edges in rel2_edges.items():
        h2t: Dict[int, Set[int]] = defaultdict(set)
        t2h: Dict[int, Set[int]] = defaultdict(set)
        for h, t in edges:
            h2t[h].add(t)
            t2h[t].add(h)
        rel2_head_to_tails[r] = h2t
        rel2_tail_to_heads[r] = t2h

    # For symmetry/anti-symmetry with PyTorch vector ops
    rel2_idx_tensors: Dict[int, torch.Tensor] = {}
    for r, edges in rel2_edges.items():
        if not edges:
            continue
        ij = torch.tensor(edges, dtype=torch.long)
        rel2_idx_tensors[r] = ij

    return rel2_edges, rel2_head_to_tails, rel2_tail_to_heads, rel2_idx_tensors


@torch.no_grad()
def detect_symmetry(
    rel2_idx: Dict[int, torch.Tensor], n_ents: int, min_ratio: float = 0.5
) -> Set[int]:
    """Return set of relations whose adjacency is sufficiently symmetric.

    min_ratio is the fraction of edges that have the reverse edge as well,
    measured on off-diagonal pairs.
    """
    sym_rels: Set[int] = set()
    for r, ij in rel2_idx.items():
        if ij.numel() == 0:
            continue
        i = ij[:, 0]
        j = ij[:, 1]
        mask = i != j
        if mask.sum().item() == 0:
            continue
        i = i[mask]
        j = j[mask]
        key = i * n_ents + j
        rev = j * n_ents + i
        # torch.isin available in recent PyTorch; fallback if needed
        try:
            hits = torch.isin(key, rev).sum().item()
        except AttributeError:
            rev_sorted, _ = torch.sort(rev)
            hits = torch.bucketize(key, rev_sorted).eq(
                torch.bucketize(key, rev_sorted, right=True)
            ).sum().item()  # rough fallback; not exact but avoids crash
        ratio = hits / key.numel()
        if ratio >= min_ratio:
            sym_rels.add(r)
    return sym_rels


@torch.no_grad()
def detect_anti_symmetry(
    rel2_idx: Dict[int, torch.Tensor], n_ents: int, max_ratio: float = 0.0
) -> Set[int]:
    """Return relations that are (near) anti-symmetric.

    max_ratio is the maximum fraction of off-diagonal reciprocal edges allowed.
    """
    anti_rels: Set[int] = set()
    for r, ij in rel2_idx.items():
        if ij.numel() == 0:
            continue
        i = ij[:, 0]
        j = ij[:, 1]
        mask = i != j
        if mask.sum().item() == 0:
            anti_rels.add(r)
            continue
        i = i[mask]
        j = j[mask]
        key = i * n_ents + j
        rev = j * n_ents + i
        try:
            overlaps = torch.isin(key, rev).sum().item()
        except AttributeError:
            rev_sorted, _ = torch.sort(rev)
            overlaps = torch.bucketize(key, rev_sorted).eq(
                torch.bucketize(key, rev_sorted, right=True)
            ).sum().item()
        ratio = overlaps / key.numel()
        if ratio <= max_ratio:
            anti_rels.add(r)
    return anti_rels


def compose_support(
    r1_h2t: Dict[int, Set[int]], r2_h2t: Dict[int, Set[int]], target_pairs: Set[Tuple[int, int]],
) -> int:
    """Compute number of (h,t) pairs generated by r1∘r2 that are present in target_pairs.

    Uses set-based expansion for scalability.
    """
    support = 0
    for h, xs in r1_h2t.items():
        for x in xs:
            ts = r2_h2t.get(x)
            if not ts:
                continue
            for t in ts:
                if (h, t) in target_pairs:
                    support += 1
    return support


def detect_composition(
    rel2_edges: Dict[int, List[Tuple[int, int]]],
    rel2_h2t: Dict[int, Dict[int, Set[int]]],
    min_support: int = 10,
    min_ratio: float = 0.1,
) -> Set[int]:
    """Return relations r for which there exist r1, r2 with substantial composition support.

    A relation r is considered compositional if max_{r1,r2} support(r1∘r2 → r)
    has at least `min_support` matches and covers at least `min_ratio` of r's edges.
    """
    comp_rels: Set[int] = set()
    rel_ids = list(rel2_edges.keys())
    # Precompute pair sets for targets
    rel2_pairset: Dict[int, Set[Tuple[int, int]]] = {
        r: set(edges) for r, edges in rel2_edges.items()
    }

    for r_target in rel_ids:
        target_pairs = rel2_pairset[r_target]
        if not target_pairs:
            continue
        tgt_size = len(target_pairs)
        best_support = 0
        for r1 in rel_ids:
            h2t1 = rel2_h2t[r1]
            if not h2t1:
                continue
            for r2 in rel_ids:
                h2t2 = rel2_h2t[r2]
                if not h2t2:
                    continue
                sup = compose_support(h2t1, h2t2, target_pairs)
                if sup > best_support:
                    best_support = sup
                # quick accept to avoid full search if clearly compositional
                if best_support >= min_support and (best_support / tgt_size) >= min_ratio:
                    break
            if best_support >= min_support and (best_support / tgt_size) >= min_ratio:
                break
        if best_support >= min_support and (best_support / tgt_size) >= min_ratio:
            comp_rels.add(r_target)
    return comp_rels


def main():
    ap = argparse.ArgumentParser(description="KG relation pattern statistics")
    ap.add_argument("--data", type=str, default=None, help="Directory with train/valid/test.txt")
    ap.add_argument("--files", nargs="*", help="Explicit triple files (whitespace h r t)")
    ap.add_argument(
        "--pykeen",
        type=str,
        default=None,
        help=(
            "PyKEEN dataset spec (e.g., 'biokg', 'BioKG', or 'pykeen.datasets.biokg:BioKG'). "
            "Requires pykeen to be installed."
        ),
    )
    ap.add_argument("--sym-min-ratio", type=float, default=0.5, help="Min symmetric edge ratio")
    ap.add_argument("--anti-max-ratio", type=float, default=0.0, help="Max reciprocal ratio for anti-symmetry")
    ap.add_argument("--comp-min-support", type=int, default=10, help="Min composition support count")
    ap.add_argument("--comp-min-ratio", type=float, default=0.1, help="Min composition coverage ratio")
    args = ap.parse_args()

    # Load triples from either PyKEEN or local files
    if args.pykeen:
        triples = read_triples_from_pykeen(args.pykeen)
    else:
        triples = read_triples(args.data, args.files)
    if not triples:
        raise SystemExit("No triples found after reading inputs")

    id_triples, ents, rels, _, _ = build_ids(triples)
    n_ents = len(ents)
    (
        rel2_edges,
        rel2_h2t,
        _rel2_t2h,
        rel2_idx,
    ) = per_relation_edges(id_triples, n_ents)

    sym_rels = detect_symmetry(rel2_idx, n_ents, args.sym_min_ratio)
    anti_rels = detect_anti_symmetry(rel2_idx, n_ents, args.anti_max_ratio)
    comp_rels = detect_composition(
        rel2_edges, rel2_h2t, args.comp_min_support, args.comp_min_ratio
    )

    if args.pykeen:
        dataset_name = args.pykeen
    else:
        dataset_name = (
            os.path.basename(os.path.abspath(args.data)) if args.data else "custom"
        )

    # Summary like the example table
    print("pattern\tdataset\tanti-symmetry\tcomposition\tsymmetry")
    print(
        f"{dataset_name}\t{len(anti_rels)}\t{len(comp_rels)}\t{len(sym_rels)}"
    )

    # Optional: verbose per-relation list
    print("\nDetails:")
    print("Symmetry:", [rels[r] for r in sorted(sym_rels)])
    print("Anti-symmetry:", [rels[r] for r in sorted(anti_rels)])
    print("Composition:", [rels[r] for r in sorted(comp_rels)])


if __name__ == "__main__":
    main()
