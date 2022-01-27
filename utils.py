from typing import *
import os
import requests

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T21')


def download(url: str, destination_folder: str):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    if "?" in filename:
        filename = filename.split('?')[0]
    file_path = os.path.join(destination_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return filename
    else:  # HTTP status code 4XX/5XX
        raise requests.HTTPError("Download failed: status code {}\n{}".format(r.status_code, r.text))