"""
I gathered 575_450 images of San Francisco under .data/raw from Mapillary.

This script backs them up to a Google Drive directory using the Google Drive API. 

The images are already on a Lambda Cloud filesystem but I wanted to back them up to GDrive also. 

"""

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Gdrive folder ID
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

# Source directory
SOURCE_DIR = ".data/raw"

def main():
  """Shows basic usage of the Drive v3 API.
  Prints the names and ids of the first 10 files the user has access to.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)

    for entry in os.scandir(SOURCE_DIR):
        if entry.is_file() and entry.name.endswith(".jpg"):
            file_path = entry.path
            media = MediaFileUpload(file_path, mimetype="image/jpeg", resumable=True)
        try:
            service.files().create(
                body={"name": entry.name, "parents": [GDRIVE_FOLDER_ID]},
                media_body=media,
                fields="id",
            ).execute()
        except HttpError as error:
            print(f"An error occurred: {error}")

if __name__ == "__main__":
  main()
