import argparse
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

parser = argparse.ArgumentParser(description='Fetch Mapillary image metadata')
parser.add_argument('--image_id', required=True, help='Mapillary image ID')
args = parser.parse_args()

token = os.getenv('MAPILLARY_CLIENT_TOKEN')
if not token:
    raise ValueError("MAPILLARY_CLIENT_TOKEN not found in .env file")

fields = 'geometry'

url = f'https://graph.mapillary.com/{args.image_id}'
params = {
    'access_token': token,
    'fields': fields
}

response = requests.get(url, params=params)
response.raise_for_status()

# "geometry"{"coordinates:[longitude, latitude]}
data = response.json()
print(json.dumps(data, indent=2))
