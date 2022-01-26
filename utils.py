from typing import *
import os
import requests

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T21')


class attr:
    """
    attr("len") @ str # check if attribute len is in string
    """
    def __init__(self, name: str):
        self.name = name

    def __matmul__(self, cls: Any) -> bool:
        return getattr(cls, self.name, None) is not None


class method:
    """
    method("append") @ [1,2,3] # check if append is a callable attribute at list class / instance
    """
    def __init__(self, name: str):
        self.name = name

    def __matmul__(self, cls: Any) -> bool:
        return callable(getattr(cls, self.name, None))


class subcls:
    """
    subcls(Tuple) @ List => False # check if class Tuple(List): ...
    """
    def __init__(self, cls: Any):
        self.t = type(cls)

    def __matmul__(self, other: Any):
        return issubclass(self.t, other)


def fassert(test_value: bool, success_value: T, msg: Optional[str] = None) -> T:
    """
    Combined assert with returning value of ok

    Example: x = fassert(isinstance(data, int), data, "expected integer")

    :param test_value: value to be asserted
    :param success_value: return value if test_value == True
    :param msg: assertion message if test_value == False
    :return: success_value / AssertionError
    """
    if msg:
        assert test_value, msg
    else:
        assert test_value
    return success_value


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