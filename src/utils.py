from typing import Optional, TypeVar, Type, Iterable, Any, Callable, Iterator, Dict, Union
from operator import lt

from functools import partial

import os

import itertools
import numpy as np
import requests

T = TypeVar("T")


def validate_isinstance(
        value: T,
        expected_type_s: Union[Type, Iterable[Type]],
        name: Optional[str] = None,
        optional: bool = False
) -> T:
    if optional and value is None:
        return value

    expected_types = expected_type_s if isinstance(expected_type_s, Iterable) else [expected_type_s, ]

    # Checking if any matches & leaving if so
    for expected_type in expected_types:
        if isinstance(value, expected_type):
            return value

    # Got here <=> is not matchable with expected types
    actual_type = type(value)
    index_repr = f"for value {value}" if name is None else f"of '{name}'"
    type_repr = " | ".join(map(str, expected_types))
    nullability = " or None" if optional else ""
    raise TypeError(f"Wrong type '{actual_type}' {index_repr}, must be an instance of '{type_repr}'{nullability}")


def flatten(
        data: Iterable[T],
        unpack_cond: Optional[Callable[[T], bool]] = None,
        drop_cond: Optional[Callable[[T], bool]] = None,
) -> Iterator[T]:
    if unpack_cond is None:
        def unpack_cond(elem: Any) -> bool:
            return isinstance(elem, Iterable)

    if drop_cond is None:
        def drop_cond(_) -> bool:
            return False

    for item in data:
        if not drop_cond(item):
            if unpack_cond(item):
                for x in item:
                    yield x
            else:
                yield item


def slice_into_chunks(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def download(url: str, destination_folder: str) -> str:
    """
    Downloads the file from url, saves to destination folder
    :param url: resource url, file extension extracted from it too
    :param destination_folder: path, where the file is saved
    :return: filename | http error fot 4XX, 5XX codes
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
    raise requests.HTTPError(f"Download failed: status code{ref.status_code}\n{ref.text}")


def split_index(length: int,
                splits: Optional[Dict[str, float]] = None,
                shuffle: bool = True) -> Dict[str, np.ndarray]:
    """
    Generates splitted and optionally shuffled index for length split into parts
    :param length: Length of generated index (ex: length=10 => indices 0, 1, ..., 9)
    :param splits: Dictionary of names and corresponding parts summing up to 1 (ex: {"train": 0.7, "test": 0.3})
    :param shuffle: Shuffling the index before splitting
    :return: Dictionary of parts with indices str -> np.array
    """
    if splits is None:
        splits = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
    else:
        vals = splits.values()
        assert all(map(lambda x: isinstance(x, float), vals)), "expected float split ratios"
        assert sum(vals) == 1, "ratio values of parts must sum up to 1"
        assert all(map(partial(lt, 0), vals)), "part ratios must be > 0"
    if shuffle:
        seed = np.random.get_state()[1][0]
        _ = np.random.random()
        generator = np.random.default_rng(seed)
        index = generator.permutation(length)
    else:
        index = np.arange(0, length)
    result = {}
    prev = 0
    for key, value in splits.items():
        value = prev + int(value * length)
        result[key] = index[prev: value]
        prev = value
    return result


def itercat(*iterables: Iterable[T]) -> Iterable[T]:
    for iterable in iterables:
        if isinstance(iterable, Iterable):
            for item in iterable:
                yield item
        else:
            yield iterable
