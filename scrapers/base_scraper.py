"""Classe abstraite de base pour tous les scrapers/loaders."""

import json
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import REQUEST_DELAY, REQUEST_TIMEOUT, MAX_RETRIES, USER_AGENT


class BaseScraper(ABC):
    """
    Fournit : gestion de session HTTP avec retry, rate limiting,
    checkpoint/resume, et logging.
    """

    def __init__(self, output_dir: Path, delay: float = REQUEST_DELAY):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = self._create_session()
        self._checkpoint_file = self.output_dir / ".checkpoint.json"

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": USER_AGENT})
        return session

    def _rate_limit(self):
        time.sleep(self.delay)

    def _fetch(self, url: str) -> requests.Response:
        self._rate_limit()
        self.logger.debug(f"Fetching: {url}")
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        response.encoding = "utf-8"
        return response

    def _fetch_json(self, url: str = None, params: dict = None) -> dict:
        self._rate_limit()
        if url:
            self.logger.debug(f"Fetching JSON: {url}")
            response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        else:
            raise ValueError("URL requise pour _fetch_json")
        response.raise_for_status()
        return response.json()

    @abstractmethod
    def collect(self) -> list[dict]:
        """Collecte les données et retourne une liste de dictionnaires."""
        pass

    def save_checkpoint(self, state: dict):
        with open(self._checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)

    def load_checkpoint(self) -> dict | None:
        if self._checkpoint_file.exists():
            with open(self._checkpoint_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def save_records(self, records: list[dict], filename: str = "data.jsonl"):
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.logger.info(f"Sauvegardé {len(records)} entrées dans {filepath}")
