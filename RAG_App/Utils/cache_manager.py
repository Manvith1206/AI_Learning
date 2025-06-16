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

    def save_to_cache(self, key: str, data_to_cache: Any) -> None:
        """Save data to a cache file."""
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data_to_cache, f)
        except Exception as e:
            print(f"Error saving to cache: {e}")

    def load_from_cache(self, key: str) -> Any:
        """Load data from a cache file if it exists."""
        cache_path = self.get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading from cache: {e}")
                return None
        return None
