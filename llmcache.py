import hashlib
import json
import sqlite3
import threading
import time
import os
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset
from loguru import logger
from pydantic import BaseModel


class CacheDB:
    """Handles database operations for storing and retrieving cache entries.

    ### Public Methods:
        * `__init__(db_path: Path)`

            Initializes the CacheDB instance with the given SQLite database path.

        * set(key: str, request_json: str, response_json: str) -> bool

            Inserts or updates a cache entry with the provided key and JSON data.

        * get_all_entries() -> dict[str, dict[str, Any]]

            Retrieves all cache entries from the database.

        * get_existing_keys() -> set[str]

            Returns a set of keys currently stored in the cache.

        * bulk_insert(items: list, update: bool = False) -> int

            Inserts multiple cache entries in bulk, optionally updating existing ones.

        * clear() -> None

            Clears all entries from the cache.
    """

    def __init__(self, db_path: Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()

        # Initialize the database
        try:
            self.initialize_db()
        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
            logger.warning(f"Please provide a different filepath or remove the file at {self.db_path}")
            raise

    def initialize_db(self) -> None:
        """Initialize SQLite database with the required table."""
        # Check if database file already exists
        if self.db_path.exists():
            self._verify_existing_db()
        else:
            self._create_new_db()

    def _verify_existing_db(self) -> None:
        """Verify and repair an existing database if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                self._ensure_table_exists(cursor)
                self._verify_schema(cursor)
                self._ensure_index_exists(cursor)
                conn.commit()
            logger.info(f"Using existing SQLite database at {self.db_path}")
        except Exception as e:
            logger.exception(f"Database corruption detected: {e}")
            raise ValueError(f"Corrupted database at {self.db_path}: {str(e)}")

    def _create_new_db(self) -> None:
        """Create a new database with the required schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                self._create_table(cursor)
                self._ensure_index_exists(cursor)
                conn.commit()
                logger.info(f"Initialized new SQLite database at {self.db_path}")
        except Exception as e:
            logger.exception(f"Failed to initialize SQLite database: {e}")
            raise

    def _ensure_table_exists(self, cursor) -> None:
        """Check if the llm_cache table exists and create it if not."""
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_cache'")
        if not cursor.fetchone():
            self._create_table(cursor)
            logger.info("Created missing llm_cache table")

    def _create_table(self, cursor) -> None:
        """Create the llm_cache table with the required schema."""
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            key TEXT PRIMARY KEY,
            request_json TEXT,
            response_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

    def _verify_schema(self, cursor) -> None:
        """Verify that the table schema has all required columns."""
        cursor.execute("PRAGMA table_info(llm_cache)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {"key", "request_json", "response_json", "created_at"}

        if not required_columns.issubset(columns):
            missing = required_columns - columns
            raise ValueError(f"Database schema is corrupted. Missing columns: {missing}")

    def _ensure_index_exists(self, cursor) -> None:
        """Create an index on the key column for faster lookups."""
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_cache_key ON llm_cache (key)")

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Get cached entry by key.

        Args:
            key: Cache key to look up

        Returns:
            Dict containing the request and response or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT request_json, response_json FROM llm_cache WHERE key = ?", (key,))
                result = cursor.fetchone()

                if result:
                    logger.debug(f"Cache hit for key: {key}. Response: {result['response_json']}")
                    return {
                        "request": result["request_json"],
                        "response": result["response_json"],
                    }

                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(self, key: str, request_json: str, response_json: str) -> bool:
        """Set entry in cache.

        Args:
            key: Cache key
            request_json: JSON string of request parameters
            response_json: JSON string of response

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT OR REPLACE INTO llm_cache (key, request_json, response_json) VALUES (?, ?, ?)",
                        (key, request_json, response_json),
                    )
                    conn.commit()
                    logger.debug(f"Saved response to cache with key: {key}, response: {response_json}")
                    return True
            except Exception as e:
                logger.error(f"Failed to save to SQLite cache: {e}")
                return False

    def get_all_entries(self) -> dict[str, dict[str, Any]]:
        """Get all cache entries from the database."""
        cache = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT key, request_json, response_json FROM llm_cache ORDER BY created_at")

                for row in cursor.fetchall():
                    cache[row["key"]] = {
                        "request": row["request_json"],
                        "response": row["response_json"],
                    }

                logger.debug(f"Retrieved {len(cache)} entries from cache database")
                return cache
        except Exception as e:
            logger.error(f"Error retrieving all cache entries: {e}")
            return {}

    def clear(self) -> bool:
        """Clear all cache entries.

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM llm_cache")
                    conn.commit()
                    logger.info("Cache cleared")
                    return True
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False

    def get_existing_keys(self) -> set:
        """Get all existing keys in the database.

        Returns:
            Set of keys
        """
        existing_keys = set()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM llm_cache")
                for row in cursor.fetchall():
                    existing_keys.add(row[0])
                return existing_keys
        except Exception as e:
            logger.error(f"Error retrieving existing keys: {e}")
            return set()

    def bulk_insert(self, items: list, update: bool = False) -> int:
        """Insert multiple items into the cache.

        Args:
            items: List of (key, request_json, response_json) tuples
            update: Whether to update existing entries

        Returns:
            Number of items inserted
        """
        count = 0
        UPDATE_OR_IGNORE = "INSERT OR REPLACE" if update else "INSERT OR IGNORE"
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.executemany(
                        f"{UPDATE_OR_IGNORE} INTO llm_cache (key, request_json, response_json) VALUES (?, ?, ?)",
                        items,
                    )
                    count = cursor.rowcount
                    conn.commit()
                return count
            except Exception as e:
                logger.error(f"Error during bulk insert: {e}")
                return 0


def parse_dataset_repo_id(repo_name: str) -> tuple[str, str, str]:
    split_count = repo_name.count(":")
    if split_count > 2:
        raise ValueError(
            f"Invalid repository name: {repo_name}. Should be in the format <repo_id>[:<config>][:<split>]"
        )
    if split_count == 0:
        repo_name = f"{repo_name}::train"
    elif split_count == 1:
        repo_name = f"{repo_name}:train"
    elif split_count == 2:
        pass

    repo_id, config, split = repo_name.split(":")
    config = config or None
    return repo_id, config, split


def load_dataset_repo(repo_name: str, **kwargs) -> Dataset:
    repo_id, config, split = parse_dataset_repo_id(repo_name)
    return load_dataset(repo_id, config=config, split=split, **kwargs)


class LLMCache:
    """
    ### Public Methods:
        * `__init__(cache_dir: str = ".", hf_repo: str | None = None, cache_sync_interval: int = 3600, reset: bool = False)`

            Initializes the LLMCache instance with the given cache directory and HF repo.

        * `get(model: str, system: str, prompt: str, response_format: dict[str, Any], temperature: float | None = None) -> Optional[dict[str, Any]]`

            Retrieves cached response if it exists.

        * `set(model: str, system: str, prompt: str, response_format: dict[str, Any], temperature: float | None, response: dict[str, Any]) -> None`

            Stores response in cache and syncs if needed.

        * `get_all_entries() -> dict[str, dict[str, Any]]`

            Retrieves all cache entries from the database.

        * `has_hf_repo() -> bool`

            Checks if the cache has an HF repo assigned.

        * `assign_hf_repo(hf_repo: str) -> None`

            Assigns an HF repo to the cache. If no repo is assigned, the cache will not sync to HF.

        * `sync_to_hf() -> None`

            Syncs the cache to the HF repo. First downloads the latest dataset from HF, then merges it with the local cache, and finally pushes the merged cache to HF.
            When in conflict (key collision), the local cache takes precedence.

        * `clear() -> None`

            Clears all cache entries.

    """

    def __init__(
        self, cache_dir: str = ".", hf_repo: str | None = None, cache_sync_interval: int = 3600, reset: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.db_path = self.cache_dir / "llm_cache.db"
        self.cache_sync_interval = cache_sync_interval
        self.last_sync_time = time.time()

        self.hf_repo: str | None = hf_repo

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Initialize CacheDB
        self.db = CacheDB(self.db_path)
        if reset:
            self.db.clear()

        # Try to load from HF dataset if available
        try:
            self._load_cache_from_hf()
        except Exception as e:
            logger.warning(f"Failed to load cache from HF dataset: {e}")

    @classmethod
    def _response_format_json(cls, response_format: BaseModel | dict) -> str:
        """Convert a response format to a JSON string."""
        # If it's a Pydantic BaseModel subclass, use its schema
        if hasattr(response_format, "model_json_schema"):
            return json.dumps(response_format.model_json_schema())

        # If it's a Pydantic model, use its dump
        elif hasattr(response_format, "model_dump_json"):
            return response_format.model_dump_json()

        if isinstance(response_format, dict):
            return json.dumps(response_format)

        return json.dumps({"value": str(response_format)})

    def _generate_key(
        self, model: str, system: str, prompt: str, response_format: Any, temperature: float | None = None
    ) -> str:
        """Generate a unique key for caching based on inputs."""
        response_format_str = self._response_format_json(response_format)
        # Include temperature in the key
        key_content = f"{model}:{system}:{prompt}:{response_format_str}"
        if temperature is not None:
            key_content += f":{temperature:.2f}"
        return hashlib.md5(key_content.encode()).hexdigest()

    def _create_request_json(
        self, model: str, system: str, prompt: str, response_format: Any, temperature: float | None
    ) -> str:
        """Create JSON string from request parameters."""
        request_data = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "response_format": self._response_format_json(response_format),
            "temperature": temperature,
        }
        return json.dumps(request_data)

    def _check_request_match(
        self,
        cached_req: dict[str, Any],
        model: str,
        system: str,
        prompt: str,
        response_format: Any,
        temperature: float | None,
    ) -> tuple[bool, str]:
        """Check if the cached request matches the new request.

        Returns:
            Tuple of (bool, str) - whether the requests match and the response format string
        """
        # Check each field and log any mismatches
        if cached_req["model"] != model:
            return False, f'model mismatch: "{cached_req["model"]}" vs "{model}"'
        if cached_req["system"] != system:
            return False, f'system mismatch: "{cached_req["system"]}" vs "{system}"'
        if cached_req["prompt"] != prompt:
            return False, f'prompt mismatch: "{cached_req["prompt"]}" vs "{prompt}"'
        resp_format_str = self._response_format_json(response_format)
        if cached_req["response_format"] != resp_format_str:
            return False, f'response_format mismatch: "{cached_req["response_format"]}" vs "{resp_format_str}"'
        if cached_req["temperature"] != temperature:
            return False, f'temperature mismatch: "{cached_req["temperature"]}" vs "{temperature}"'
        return True, ""

    def get(
        self, model: str, system: str, prompt: str, response_format: dict[str, Any], temperature: float | None = None
    ) -> Optional[dict[str, Any]]:
        """Retrieve cached response if it exists."""
        key = self._generate_key(model, system, prompt, response_format, temperature)
        result = self.db.get(key)

        if not result:
            return None
        cached_req = json.loads(result["request"])
        match, reason = self._check_request_match(cached_req, model, system, prompt, response_format, temperature)
        if not match:
            logger.warning(f"Cached request does not match new request for key: {key}. Reason: {reason}")
            return None

        return json.loads(result["response"])

    def set(
        self,
        model: str,
        system: str,
        prompt: str,
        response_format: dict[str, Any],
        temperature: float | None,
        response: dict[str, Any],
    ) -> None:
        """Store response in cache and sync if needed."""
        key = self._generate_key(model, system, prompt, response_format, temperature)
        request_json = self._create_request_json(model, system, prompt, response_format, temperature)
        response_json = json.dumps(response)

        success = self.db.set(key, request_json, response_json)

        # Check if we should sync to HF
        if success and self.hf_repo and (time.time() - self.last_sync_time > self.cache_sync_interval):
            try:
                self.sync_to_hf()
                self.last_sync_time = time.time()
            except Exception as e:
                logger.error(f"Failed to sync cache to HF dataset: {e}")

    def get_all_entries(self) -> dict[str, dict[str, Any]]:
        """Retrieve all cache entries from the database."""
        cache = self.db.get_all_entries()
        entries = {}
        for key, entry in cache.items():
            request = json.loads(entry["request"])
            response = json.loads(entry["response"])
            entries[key] = {"request": request, "response": response}
        return entries

    def has_hf_repo(self) -> bool:
        """Determine if the cache has an HF repo."""
        return self.hf_repo is not None

    def assign_hf_repo(self, hf_repo: str) -> None:
        """Assign an HF repo to the cache."""
        self.hf_repo = hf_repo

    def _load_cache_from_hf(self) -> None:
        """Load cache from HF dataset if it exists and merge with local cache."""
        if not self.hf_repo:
            return

        try:
            # Check for new commits before loading the dataset
            dataset = load_dataset_repo(self.hf_repo, download_mode="force_redownload")

            existing_keys = self.db.get_existing_keys()

            logger.info(f"Found {len(dataset)} items in HF dataset. Existing keys: {len(existing_keys)}")

            # Prepare batch items for insertion
            items_to_insert = []
            for item in dataset:
                key = item["key"]
                # Only update if not in local cache to prioritize local changes
                if key in existing_keys:
                    continue
                # Create request JSON
                request_data = {
                    "model": item["model"],
                    "system": item["system"],
                    "prompt": item["prompt"],
                    "temperature": item["temperature"],
                    "response_format": item["response_format"],
                }

                items_to_insert.append(
                    (
                        key,
                        json.dumps(request_data),
                        item["response"],  # This is already a JSON string
                    )
                )

            # Bulk insert new items
            if items_to_insert:
                inserted_count = self.db.bulk_insert(items_to_insert)
                logger.info(f"Merged {inserted_count} items from HF dataset into SQLite cache")
            else:
                logger.info("No new items to merge from HF dataset")
        except Exception as e:
            logger.warning(f"Could not load cache from HF dataset: {e}")

    def sync_to_hf(self) -> None:
        """Sync cache to HF dataset."""
        if not self.hf_repo:
            return

        self._load_cache_from_hf()

        # Get all entries from the database
        cache = self.db.get_all_entries()

        # Convert cache to dataset format
        entries = []
        for key, entry in cache.items():
            request = json.loads(entry["request"])
            response_str = entry["response"]
            entries.append(
                {
                    "key": key,
                    "model": request["model"],
                    "system": request["system"],
                    "prompt": request["prompt"],
                    "response_format": request["response_format"],
                    "temperature": request["temperature"],
                    "response": response_str,
                }
            )

        if not entries:
            logger.info("No entries to sync to HF")
            return

        # Create and push dataset
        dataset = Dataset.from_list(entries)
        repo_id, config_name, split = parse_dataset_repo_id(self.hf_repo)
        config_name = config_name or "default"
        token = os.environ.get("HF_TOKEN_WRITE") or os.environ.get("HF_TOKEN_READ") or os.environ.get("HF_TOKEN")
        dataset.push_to_hub(repo_id, config_name=config_name, split=split, private=True, token=token)
        logger.info(f"Finished syncing {len(cache)} cached items to HF dataset {repo_id}")

    def clear(self) -> None:
        """Clear all cache entries."""
        self.db.clear()
