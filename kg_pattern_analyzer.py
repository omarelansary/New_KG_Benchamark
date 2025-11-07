# kg_pattern.py
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

try:
    from pykeen.datasets import get_dataset
    HAVE_PYKEEN = True
except Exception:
    HAVE_PYKEEN = False


class KGPatternAnalyzer:
    """Analyze a knowledge graph for common relation patterns (symmetry, anti-symmetry, composition)."""

    def __init__(
        self,
        sym_threshold: float = 0.8,
        anti_threshold: float = 0.01,
        comp_conf: float = 0.8,
        comp_support: int = 50,
        max_pairs: int = 5000,
    ):
        self.sym_threshold = sym_threshold
        self.anti_threshold = anti_threshold
        self.comp_conf = comp_conf
        self.comp_support = comp_support
        self.max_pairs = max_pairs

    # -----------------------
    # Data loading
    # -----------------------
    def load_triples_from_folder(self, path: Path):
        """Load triples from a local folder containing train.tsv."""
        train_path = path / "train.tsv"
        if not train_path.exists():
            raise FileNotFoundError(f"Expected {train_path} with three columns: head\trelation\ttail")

        triples = []
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    raise ValueError(f"Bad line: {line}")
                h, r, t = parts
                triples.append((h, r, t))
        return triples

    def load_triples_from_pykeen(self, dataset_name: str):
        """Load triples from a PyKEEN dataset by name."""
        if not HAVE_PYKEEN:
            raise RuntimeError("PyKEEN not installed. Run `pip install pykeen`.")
        ds = get_dataset(dataset=dataset_name)
        # df = ds.training.to_df()
        # triples = list(map(tuple, df[["head", "relation", "tail"]].values.tolist()))
        entity_labels = list(ds.training.entity_to_id.keys())
        relation_labels = list(ds.training.relation_to_id.keys())
        triples = [
            (entity_labels[h], relation_labels[r], entity_labels[t])
            for h, r, t in ds.training.mapped_triples.tolist()
        ]
        return triples

    # -----------------------
    # Pattern detection
    # -----------------------
    @staticmethod
    def build_indexes(triples):
        edges_by_rel = defaultdict(set)
        tails_by_rel_head = defaultdict(set)
        heads_by_rel_tail = defaultdict(set)

        for h, r, t in triples:
            edges_by_rel[r].add((h, t))
            tails_by_rel_head[(r, h)].add(t)
            heads_by_rel_tail[(r, t)].add(h)

        rev_edges_by_rel = {r: {(t, h) for (h, t) in s} for r, s in edges_by_rel.items()}
        return edges_by_rel, rev_edges_by_rel, tails_by_rel_head, heads_by_rel_tail

    def symmetry_stats(self, edges_by_rel, rev_edges_by_rel):
        num_sym = 0
        num_anti = 0
        for r, edges in edges_by_rel.items():
            if not edges:
                continue
            non_self = [(h, t) for (h, t) in edges if h != t]
            if not non_self:
                continue
            rev_fraction = sum(1 for (h, t) in non_self if (t, h) in edges) / len(non_self)

            if rev_fraction >= self.sym_threshold:
                num_sym += 1
            elif rev_fraction <= self.anti_threshold:
                num_anti += 1
        return num_anti, num_sym

    def composition_stats(self, edges_by_rel):
        relations = list(edges_by_rel.keys())
        rel_pairs = list(product(relations, relations))
        if len(rel_pairs) > self.max_pairs:
            random.shuffle(rel_pairs)
            rel_pairs = rel_pairs[:self.max_pairs]

        composed_relations = set()
        for r1, r2 in rel_pairs:
            m_to_t = defaultdict(set)
            for (m, t) in edges_by_rel[r2]:
                m_to_t[m].add(t)

            composed_edges = set()
            possible_m = set(x for _, x in edges_by_rel[r1]) & set(m_to_t.keys())
            if not possible_m:
                continue

            for (h, m) in edges_by_rel[r1]:
                if m not in m_to_t:
                    continue
                for t in m_to_t[m]:
                    composed_edges.add((h, t))

            support = len(composed_edges)
            if support < self.comp_support:
                continue

            for r3, edges_r3 in edges_by_rel.items():
                overlap = len(composed_edges & edges_r3)
                conf = overlap / support
                if conf >= self.comp_conf:
                    composed_relations.add(r3)
        return len(composed_relations)

    # -----------------------
    # Master function
    # -----------------------
    def analyze(self, triples):
        edges_by_rel, rev_edges_by_rel, _, _ = self.build_indexes(triples)
        anti, sym = self.symmetry_stats(edges_by_rel, rev_edges_by_rel)
        comp = self.composition_stats(edges_by_rel)
        return {
            "anti-symmetry": anti,
            "composition": comp,
            "symmetry": sym,
        }
