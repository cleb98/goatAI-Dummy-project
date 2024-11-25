import os
import io
import zipfile
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload

# Setup the Drive v3 API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES).run_local_server(port=0)
service = build('drive', 'v3', credentials=creds)

def download_and_unzip(file_id, destination_folder):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print("Download progress: {0}".format(status.progress() * 100))

    fh.seek(0)
    with zipfile.ZipFile(fh, "r") as zip_ref:
        zip_ref.extractall(destination_folder)
    print(f"File unzipped successfully to {destination_folder}")

# Provide the file ID and the destination folder
file_id = '11wCoSgY-QvJw147HieBv7E4sIEaLNEoW'  # Change to your actual file ID
destination_folder = '/path/to/destination/folder'  # Change to your actual destination

download_and_unzip(file_id, destination_folder)
