import hashlib
import os
import pickle
from typing import Any, Dict, Optional

class CacheManager:
    """Manages caching of processed vector stores to avoid re-computation."""

    def __init__(self, cache_dir: str = "cache"):
        """Initializes the CacheManager, ensuring the cache directory exists."""
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def generate_cache_key(self, file_content: bytes, params: Dict[str, Any]) -> str:
        """Generates a unique SHA256 hash based on file content and parameters."""
        hasher = hashlib.sha256()
        hasher.update(file_content)

        # Add sorted parameter items to the hash to ensure consistency
        for key, value in sorted(params.items()):
            hasher.update(str(key).encode())
            hasher.update(str(value).encode())

        return hasher.hexdigest()

    def get_cache_path(self, key: str) -> str:
        """Constructs the full path for a given cache key."""
        return os.path.join(self.cache_dir, f"{key}.pkl")

    def save_to_cache(self, key: str, data: Any) -> None:
        """Saves data to a pickle file in the cache directory."""
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Successfully saved data to cache: {cache_path}")
        except (pickle.PicklingError, IOError) as e:
            print(f"Error saving to cache file {cache_path}: {e}")

    def load_from_cache(self, key: str) -> Optional[Any]:
        """Loads data from a pickle file in the cache directory if it exists."""
        cache_path = self.get_cache_path(key)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"Successfully loaded data from cache: {cache_path}")
            return data
        except (pickle.UnpicklingError, IOError, EOFError) as e:
            print(f"Error loading from cache file {cache_path}: {e}")
            # If the cache file is corrupt, it's safer to remove it
            os.remove(cache_path)
            return None
