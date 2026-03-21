import subprocess
import os

RAW = """
""".strip()
         
pairs = []
for line in RAW.splitlines():
    parts = line.strip().split("|")
    pairs.append((parts[3], parts[4]))

KEY = "/Users/steve/documents/files/code/cityzero/wip/private/vast/key"
PORT = "PORT"
HOST = "root@HOST"
REMOTE_DIR = "/workspace/images/"
LOCAL_BASE = os.path.join(os.getcwd(), "cityzero_pairs")

for i, (img_a, img_b) in enumerate(pairs, start=1):
    pair_dir = os.path.join(LOCAL_BASE, f"pair{i}")
    os.makedirs(pair_dir, exist_ok=True)

    for img in (img_a, img_b):
        cmd = [
            "scp",
            "-i", KEY,
            "-P", PORT,
            f"{HOST}:{REMOTE_DIR}{img}",
            pair_dir
        ]
        print(f"[pair{i}] Fetching {img}...")
        subprocess.run(cmd, check=True)

print(f"\nDone. Pairs saved to {LOCAL_BASE}")