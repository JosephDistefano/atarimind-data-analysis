import collections
import pickle
import gzip

def nested_dict():
    return collections.defaultdict(nested_dict)


def save_dataset(path, dataset, save):
    """save the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    # if save:
        # dd.io.save(path, dataset, compression=('blosc', 5))
    with gzip.open(path, 'wb') as f:
        pickle.dump(dataset, f)
    return None

def read_dataset(path):
    """Read the dataset.

    Parameters
    ----------
    path : str
        path to save.
    dataset : dataset
        pytorch dataset.
    save : Bool

    """
    with gzip.open(path, 'rb') as file:
        data = pickle.load(file)
    return data
