"""
Utilities for dowloading the data and other things
"""

import os
import requests


def download(url: str, destination_folder: str) -> str:
    """
    Downloads the file from url, saves to destination folder
    :param url: resource url, file extension extracted from it too
    :param destination_folder: path, where the file is saved
    :return: filename
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # create folder if it does not exist
    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    if "?" in filename:
        filename = filename.split('?')[0]
    file_path = os.path.join(destination_folder, filename)
    ref = requests.get(url, stream=True)
    if ref.ok:
        with open(file_path, 'wb') as file:
            for chunk in ref.iter_content(chunk_size=1024 * 8):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
        return filename
    else:  # HTTP status code 4XX/5XX
        raise requests.HTTPError(f"Download failed: status code{ref.status_code}\n{ref.text}")
