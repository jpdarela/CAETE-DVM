import pickle
import bz2
import sys
import numpy as np

def read_pickle(file_path):
    """Read a compressed pickle file and save its contents to separate text files.

    Args:
        file_path (str): Path to the compressed pickle file.

    Returns:
        dict: The data loaded from the pickle file.
    """
    with bz2.BZ2File(file_path, 'rb') as f:
        data = pickle.load(f)
    # Save each value to a separate text file named after the respective key
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            v = np.array([v])  # Transform scalar into one-element array
        np.savetxt(f"{k}.txt", v, fmt='%s')
    return data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_pickle.py <pickle_file>")
        sys.exit(1)
    dt = read_pickle(sys.argv[1]) 