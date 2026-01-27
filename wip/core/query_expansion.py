"""
Query expansion for COLMAP match graph using transitive matching.

This script iteratively expands match connectivity by:
1. Finding high-quality existing matches (A↔B, B↔C)
2. Proposing transitive candidates (A↔C)
3. Verifying proposals through COLMAP's matches_importer
4. Repeating for N rounds to densify the match graph

The expansion helps bridge spatial partitions and improve reconstruction
connectivity without exhaustive pairwise matching. It helps reconstruct the scene more densely as parts that were far away can still be reconnected as necessary.

IMPORTANT: Note that featurematching between blocks decreases exponentially, specially as proposed blocks become larger. Upon first running this script you may see 40s-30s load times for block processing. Expect it to average out at 3s per block.

"""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Set, List, Tuple, Dict

# Decodes Image ID Pairs, stored as 32-bit signed ints
PAIR_ID_MULTIPLIER = 2147483647


def load_image_id_to_name(db_path: str) -> Dict[int, str]:
    """
    Load image_id -> name mapping from the images table.
    
    Returns dictionary mapping image_id to image filename.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT image_id, name FROM images")
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database query failed: {e}")
        conn.close()
        sys.exit(1)
    
    id_to_name = {image_id: name for image_id, name in rows}
    conn.close()
    print(f"Loaded {len(id_to_name)} image ID to name mappings")
    return id_to_name


def load_existing_pairs(db_path: str, min_matches: int) -> Set[Tuple[int, int]]:
    """
    Load existing image pairs from two_view_geometries with sufficient inliers.
    
    Returns set of (img1, img2) tuples where img1 < img2 and rows >= min_matches.
    """
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


def write_pair_file(pairs: List[Tuple[int, int]], output_path: Path, id_to_name: Dict[int, str]) -> None:
    """
    Write image pairs to text file (one pair per line, space-separated).
    
    Writes image filenames instead of IDs, as required by COLMAP's matches_importer.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    missing_names = []
    for img1_id, img2_id in pairs:
        if img1_id not in id_to_name:
            missing_names.append(img1_id)
        if img2_id not in id_to_name:
            missing_names.append(img2_id)
    
    if missing_names:
        unique_missing = set(missing_names)
        print(f"Error: Missing names for {len(unique_missing)} image ID(s): {sorted(unique_missing)[:10]}")
        sys.exit(1)
    
    with open(output_path, 'w') as f:
        for img1_id, img2_id in pairs:
            img1_name = id_to_name[img1_id]
            img2_name = id_to_name[img2_id]
            f.write(f"{img1_name} {img2_name}\n")
    
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
                       min_matches: int,
                       id_to_name: Dict[int, str]) -> int:
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
    write_pair_file(new_proposals, pair_file, id_to_name)
    
    # Verify proposals with COLMAP
    run_matches_importer(database_path, str(pair_file))
    
    return len(new_proposals)


def cleanup_temp_files(output_dir: Path) -> None:
    """
    Delete all temporary .txt files created during expansion rounds.
    
    Removes all files matching expansion_round_*.txt pattern in output_dir.
    """
    if not output_dir.exists():
        return
    
    deleted_count = 0
    for temp_file in output_dir.glob("expansion_round_*.txt"):
        try:
            temp_file.unlink()
            deleted_count += 1
        except OSError as e:
            print(f"Warning: Failed to delete {temp_file}: {e}")
    
    if deleted_count > 0:
        print(f"Deleted {deleted_count} temporary .txt file(s)")


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
    parser.add_argument(
        "--temp_files",
        action="store_true",
        default=False,
        help="Delete temporary .txt files after completion (default: False)"
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
    
    # Load image ID to name mapping
    id_to_name = load_image_id_to_name(str(db_path))
    
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
            args.min_matches,
            id_to_name
        )
        
        if proposals_count == 0:
            print(f"\nExpansion converged after {round_num - 1} rounds")
            break
    
    print("\n" + "="*60)
    print("Query expansion complete!")
    print("="*60)
    
    # Cleanup temporary files if requested
    if args.temp_files:
        cleanup_temp_files(output_dir)


if __name__ == "__main__":
    main()
