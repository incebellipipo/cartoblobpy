import os
import pickle

def get_map_path(map_name: str) -> str:
    """
    Get the absolute path to an example map file in this assets package.

    :param map_name: File name of the map (e.g., ``"map012.yaml"``).
    :type map_name: str
    :returns: Absolute path to the map file.
    :rtype: str
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), map_name)

def list_example_maps() -> list:
    """
    List all available example map files in the assets directory.

    :returns: File names of maps (``.yaml``) present in the assets directory.
    :rtype: list[str]
    """
    assets_dir = os.path.dirname(os.path.realpath(__file__))
    return [f for f in os.listdir(assets_dir) if f.endswith('.yaml')]

def save_result(result, filename: str):
    """
    Save a Python object as a pickle file under ``assets/results``.

    :param result: Python object to persist via ``pickle``.
    :type result: Any
    :param filename: Target file name (e.g., ``"run.pkl"``).
    :type filename: str
    :returns: Absolute path to the saved result file.
    :rtype: str
    :raises OSError: If the file cannot be written.
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    return filepath

def load_result(filename: str):
    """
    Load a pickled result object from ``assets/results``.

    :param filename: File name to load (e.g., ``"run.pkl"``).
    :type filename: str
    :returns: The deserialized Python object.
    :rtype: Any
    :raises FileNotFoundError: If the file does not exist.
    :raises pickle.UnpicklingError: If the file content is not a valid pickle.
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    return result

def list_available_results():
    """
    List all available result files in ``assets/results`` directory.

    :returns: File names ending with ``.pkl`` under ``assets/results``.
    :rtype: list[str]
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.exists(results_dir):
        return []
    return [f for f in os.listdir(results_dir) if f.endswith('.pkl')]