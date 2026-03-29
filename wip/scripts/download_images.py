import os
import boto3

KEY_ID = ""
APP_KEY = ""

IMGS = (
)

out = os.path.join(os.path.dirname(__file__), "temp_imgs")
os.makedirs(out, exist_ok=True)

if not KEY_ID or not APP_KEY:
    print("Set KEY_ID and APP_KEY in this file.")
else:
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.us-west-004.backblazeb2.com",
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=APP_KEY,
        region_name="us-west-004",
    )
    for name in IMGS:
        dest = os.path.join(out, name)
        print(name, end=" ... ")
        s3.download_file("cityzero-sf-backup", f"images/{name}", dest)
        print("ok")
    print("→", out)
