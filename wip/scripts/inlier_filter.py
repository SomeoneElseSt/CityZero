import sqlite3
import struct
import math
from pathlib import Path

DB_PATH = Path("database/database.db")

def decode_position(blob):
    lat, lon, alt = struct.unpack('<ddd', blob)
    return lat, lon

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT corr_data_id, position FROM pose_priors")
positions = {row[0]: decode_position(row[1]) for row in cur.fetchall()}

cur.execute("""
    SELECT
        tvg.rows AS num_inliers,
        tvg.pair_id / 2147483647 AS id_a,
        tvg.pair_id % 2147483647 AS id_b,
        i1.name AS name_a,
        i2.name AS name_b
    FROM two_view_geometries tvg
    JOIN images i1 ON i1.image_id = tvg.pair_id / 2147483647
    JOIN images i2 ON i2.image_id = tvg.pair_id % 2147483647
    WHERE tvg.rows >= 100 AND tvg.rows < 200
""")

results = []
for num_inliers, id_a, id_b, name_a, name_b in cur.fetchall():
    pos_a = positions.get(id_a)
    pos_b = positions.get(id_b)
    if pos_a is None or pos_b is None:
        continue
    dist = haversine(*pos_a, *pos_b)
    if dist <= 50:
        results.append((dist, num_inliers, name_a, name_b))

conn.close()

results.sort(key=lambda x: x[0], reverse=True)

for dist, inliers, name_a, name_b in results[:10]:
    print(f"# dist={dist:.1f}m  inliers={inliers}")
    print(f"catimg {name_a} && catimg {name_b}")
    print()