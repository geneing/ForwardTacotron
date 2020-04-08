import pickle
from pathlib import Path
from typing import Union

def get_files(path: Union[str, Path], extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def pickle_binary(data: object, file: str):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: str):
    with open(file, 'rb') as f:
        return pickle.load(f)
