"""Small HTTP helper: disk cache, retries with backoff, and a monthly quota counter."""
import hashlib
import json
import time
from datetime import datetime

import requests

from config import CACHE

_QUOTA_FILE = CACHE / "quota.json"


def _count(provider):
    q = json.loads(_QUOTA_FILE.read_text()) if _QUOTA_FILE.exists() else {}
    key = f"{provider}:{datetime.now():%Y-%m}"
    q[key] = q.get(key, 0) + 1
    _QUOTA_FILE.write_text(json.dumps(q, indent=1))
    return q[key]


def used(provider):
    q = json.loads(_QUOTA_FILE.read_text()) if _QUOTA_FILE.exists() else {}
    return q.get(f"{provider}:{datetime.now():%Y-%m}", 0)


def get_json(url, *, provider, params=None, headers=None, ttl_s=3600, retries=3,
             method="GET", data=None, timeout=30):
    """Cached request. ttl_s=None caches forever (use for historical data)."""
    key = hashlib.sha1(json.dumps([method, url, params, data], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{provider}_{key}.json"
    if path.exists() and (ttl_s is None or time.time() - path.stat().st_mtime < ttl_s):
        return json.loads(path.read_text())
    last = None
    for attempt in range(retries):
        try:
            r = requests.request(method, url, params=params, headers=headers, data=data, timeout=timeout)
            _count(provider)
            if r.status_code == 429:
                raise RuntimeError(f"{provider}: rate limit / quota exceeded (429)")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            js = r.json()
            path.write_text(json.dumps(js))
            return js
        except RuntimeError:
            raise
        except Exception as e:                      # network error -> retry with backoff
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{provider}: request failed after {retries} tries: {last}")
