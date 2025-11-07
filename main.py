# main.py
from kg_pattern_analyzer import KGPatternAnalyzer
from pathlib import Path

def main():
    analyzer = KGPatternAnalyzer(sym_threshold=0.8, anti_threshold=0.01)

    datasets = ["pharmkg", "primekg"]
    print("pattern\tdataset\tanti-symmetry\tcomposition\tsymmetry")

    for name in datasets:
        triples = analyzer.load_triples_from_pykeen(name)
        stats = analyzer.analyze(triples)
        print(f"{name}\t{stats['anti-symmetry']}\t{stats['composition']}\t{stats['symmetry']}")

    # If you have a local KG folder:
    # local_path = Path("data/my_kg")
    # triples = analyzer.load_triples_from_folder(local_path)
    # stats = analyzer.analyze(triples)
    # print(stats)

if __name__ == "__main__":
    main()
