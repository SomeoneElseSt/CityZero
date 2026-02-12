### How to get set up on AMD GPU Clusters

SSH'ing in:
```
ssh -i amd.pem root@<your-ip>
```

This will ask for a password stored locally under the wip folder. 

Once authenticated, scp ```build_colmap_amd.py```:

```
scp -i amd.pem build_colmap_amd.py ubuntu@<your-ip>:/home/
```

and ```mapper_tuned_amd.py```:

```
scp -i amd.pem mapper_tuned_amd.py ubuntu@<your-ip>:/home/
```

This will also ask for a password. 

To download the database and images, configure rclone and download the ```images```, ```mapper_db```, ```all_images.txt```.

1. Download images:

```
rclone copy b2:cityzero-sf-backup/images ./images --transfers 32 --checkers 64 --fast-list --buffer-size 64M -P
```

2. Download mapper_db:
```
rclone copy b2:cityzero-sf-backup/mapper_db ./mapper_db --multi-thread-streams 16 --multi-thread-cutoff 64M --buffer-size 64M -P
```

3. Download all_images.txt: 
```
rclone copy b2:cityzero-sf-backup/image_boxes/all_images.txt ./ -P
```

4. Download last snapshot binaries: 
```
rclone copy b2:cityzero-sf-backup/snapshots/<snapshot-N> ./initial_reconstruction/ --transfers 32 --checkers 64 --fast-list --buffer-size 64M -P

```
or upload them locally: 

```
 scp -r -i amd.pem outputs/snapshots/snapshots_mapper_v<N>/0/ root@<IP>:/home/initial_reconstruction/

```
5. Make a 'snapshots' and 'outputs' directories
```
mkdir -p snapshots && mkdir -p outputs
```

6. And run the mapper on a new tmux session from the ~/home directory:  
```
python3 mapper_tuned_amd.py --database_path mapper_db/database.db --image_path images/ --output_path outputs/ --image_list_path all_images.txt --snapshot_path snapshots/ --input_path initial_reconstruction/ --snapshot_frames_freq 100 

```

When done, upload most recent snapshot to cityzero-sf-backup/snapshots/:
```
rclone copy ./snapshots/<snapshot-N> b2:cityzero-sf-backup/snapshots/<snapshot-N> --transfers 32 --checkers 64 -P

```
