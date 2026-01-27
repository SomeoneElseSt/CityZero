"""
Query expansion for COLMAP match graph using transitive matching.

This script iteratively expands match connectivity by:
1. Finding high-quality existing matches (A↔B, B↔C)
2. Proposing transitive candidates (A↔C)
3. Verifying proposals through COLMAP's matches_importer
4. Repeating for N rounds to densify the match graph

The expansion helps bridge spatial partitions and improve reconstruction
connectivity without exhaustive pairwise matching.
"""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, List, Tuple

# Constants
PAIR_ID_MULTIPLIER = 2147483647


def load_existing_pairs(db_path: str, min_matches: int) -> Set[Tuple[int, int]]:
    """
    Load existing image pairs from two_view_geometries with sufficient inliers.
    
    Returns set of (img1, img2) tuples where img1 < img2 and rows >= min_matches.
    """
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows >= ?", (min_matches,))
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database query failed: {e}")
        conn.close()
        sys.exit(1)
    
    existing_pairs = set()
    for pair_id, inliers in rows:
        img1 = (pair_id - pair_id % PAIR_ID_MULTIPLIER) // PAIR_ID_MULTIPLIER
        img2 = pair_id % PAIR_ID_MULTIPLIER
        # Canonical ordering: smaller ID first
        existing_pairs.add((min(img1, img2), max(img1, img2)))
    
    conn.close()
    print(f"Loaded {len(existing_pairs)} existing pairs with >={min_matches} matches")
    return existing_pairs


def build_adjacency_graph(pairs: Set[Tuple[int, int]]) -> dict:
    """Build adjacency list from image pairs."""
    graph = defaultdict(set)
    for img1, img2 in pairs:
        graph[img1].add(img2)
        graph[img2].add(img1)
    return graph


def find_transitive_candidates(existing_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Find transitive match candidates: if A↔B and B↔C exist, propose A↔C.
    
    Returns list of proposed (img1, img2) pairs with canonical ordering.
    """
    print("Building adjacency graph...")
    graph = build_adjacency_graph(existing_pairs)
    
    print("Finding transitive candidates...")
    candidates = set()
    
    # For each intermediate node B
    for node_b in graph:
        neighbors = list(graph[node_b])
        # For each pair of B's neighbors (A, C)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                node_a = neighbors[i]
                node_c = neighbors[j]
                # Propose A↔C with canonical ordering
                pair = (min(node_a, node_c), max(node_a, node_c))
                candidates.add(pair)
    
    print(f"Found {len(candidates)} transitive candidates")
    return list(candidates)


def filter_existing_pairs(candidates: List[Tuple[int, int]], 
                         existing_pairs: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove candidates that already exist in the match graph."""
    filtered = [pair for pair in candidates if pair not in existing_pairs]
    print(f"After filtering existing pairs: {len(filtered)} new proposals")
    return filtered


def write_pair_file(pairs: List[Tuple[int, int]], output_path: Path) -> None:
    """Write image pairs to text file (one pair per line, space-separated)."""
    if not pairs:
        print(f"No pairs to write to {output_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")
    
    print(f"Wrote {len(pairs)} pairs to {output_path}")


def run_matches_importer(database_path: str, match_list_path: str) -> None:
    """Run COLMAP matches_importer to verify proposed pairs."""
    print(f"\nRunning matches_importer on {match_list_path}")
    print("IMPORTANT: Large databases may have 5-15 min delay before processing starts")
    
    cmd = [
        "colmap",
        "matches_importer",
        "--database_path", database_path,
        "--match_list_path", match_list_path,
        "--match_type", "pairs",
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "-1",
        "--FeatureMatching.num_threads", "-1",
        "--FeatureMatching.guided_matching", "1",
    ]
    
    try:
        subprocess.run(cmd, check=True, text=True)
        print("matches_importer completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"matches_importer failed with exit code: {e.returncode}")
        sys.exit(1)


def run_expansion_round(round_num: int, 
                       database_path: str, 
                       output_dir: Path,
                       min_matches: int) -> int:
    """
    Execute one round of query expansion.
    
    Returns number of new proposals generated.
    """
    print(f"\n{'='*60}")
    print(f"EXPANSION ROUND {round_num}")
    print(f"{'='*60}")
    
    # Load current state of match graph
    existing_pairs = load_existing_pairs(database_path, min_matches)
    
    # Find transitive candidates
    candidates = find_transitive_candidates(existing_pairs)
    
    # Filter out already-matched pairs
    new_proposals = filter_existing_pairs(candidates, existing_pairs)
    
    if not new_proposals:
        print(f"No new proposals in round {round_num}, stopping expansion")
        return 0
    
    # Write proposals to file
    pair_file = output_dir / f"expansion_round_{round_num}.txt"
    write_pair_file(new_proposals, pair_file)
    
    # Verify proposals with COLMAP
    run_matches_importer(database_path, str(pair_file))
    
    return len(new_proposals)


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively expand COLMAP match graph using transitive matching"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to COLMAP database file (database.db)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write expansion_round_N.txt files"
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=30,
        help="Minimum inlier matches for propagation (default: 30)"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=4,
        help="Number of expansion rounds (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    db_path = Path(args.database_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)
    
    if not str(db_path).endswith('.db'):
        print("Database path must end with .db")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    print(f"\nQuery Expansion Configuration:")
    print(f"  Database: {db_path}")
    print(f"  Output directory: {output_dir}")
    print(f"  Min matches threshold: {args.min_matches}")
    print(f"  Number of rounds: {args.num_rounds}")
    
    # Run expansion rounds
    for round_num in range(1, args.num_rounds + 1):
        proposals_count = run_expansion_round(
            round_num, 
            str(db_path), 
            output_dir,
            args.min_matches
        )
        
        if proposals_count == 0:
            print(f"\nExpansion converged after {round_num - 1} rounds")
            break
    
    print("\n" + "="*60)
    print("Query expansion complete!")
    print("="*60)


if __name__ == "__main__":
    main()
