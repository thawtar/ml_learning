from config import data_conf

# File to load data
from logging import Logger
import pandas as pd
from io import StringIO

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import yaml

# Initialize logger
logger = Logger(__name__)




class DataLoader:
    def __init__(self,file_path=None):
        
        self.file_path = file_path
        self.data = None

    def load_data(self):
        print("Loading data...")
        source = data_conf['source']
        if source=='local':
            if not self.file_path:
                data_path = data_conf['local']['data_path']
            else:
                data_path = self.file_path
            logger.info(f"Loading data from local path: {data_path}")
            data = pd.read_csv(data_path)
            return data
        elif source=='gdrive':
            file_id = data_conf['gdrive']['file_id']
            credentials_path = data_conf['gdrive']['credentials_path']
            token_path = data_conf['gdrive']['token_path']
            logger.info(f"Loading data from Google Drive with file ID: {file_id}")

            gauth = GoogleAuth()
            gauth.LoadCredentialsFile(token_path)
            if gauth.credentials is None:
                gauth.LocalWebserverAuth()
            elif gauth.access_token_expired:
                gauth.Refresh()
            else:
                gauth.Authorize()
            gauth.SaveCredentialsFile(token_path)

            drive = GoogleDrive(gauth)
            downloaded = drive.CreateFile({'id': file_id})
            content = downloaded.GetContentString()
            data = pd.read_csv(StringIO(content))
            return data
        else:
            logger.error(f"Unknown data source: {source}")
            raise ValueError(f"Unknown data source: {source}")

def main():
    print("Testing DataLoader...")
    data_loader = DataLoader()
    data = data_loader.load_data()
    print("Data loaded successfully:")
    print(data.head())

if __name__ == "__main__":
    main()