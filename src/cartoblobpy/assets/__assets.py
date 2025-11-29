import os
import pickle

def get_map_path(map_name: str) -> str:
    """
    Get the path to the example map file.
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), map_name)

def list_example_maps() -> list:
    """
    List all available example map files in the assets directory.
    """
    assets_dir = os.path.dirname(os.path.realpath(__file__))
    return [f for f in os.listdir(assets_dir) if f.endswith('.yaml')]

def save_result(result, filename: str):
    """
    Save the result object to assets/results directory.
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)

def load_result(filename: str):
    """
    Load the result object from assets/results directory.
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    return result

def list_available_results():
    """
    List all available result files in assets/results directory.
    """
    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
    if not os.path.exists(results_dir):
        return []
    return [f for f in os.listdir(results_dir) if f.endswith('.pkl')]