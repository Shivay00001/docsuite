import json
from pathlib import Path
from datetime import datetime
from typing import Dict

class UsageTracker:
    """
    Tracks API and processing usage locally.
    Persists to a secure local file (could be SQLite/Encrypted JSON).
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_usage()

    def _load_usage(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self.usage_data = json.load(f)
            except:
                self.usage_data = {"total_pages": 0, "history": []}
        else:
            self.usage_data = {"total_pages": 0, "history": []}

    def _save_usage(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.usage_data, f)

    def track_process(self, pages_count: int, doc_type: str = "unknown"):
        """Record a document processing event."""
        self.usage_data["total_pages"] += pages_count
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "pages": pages_count,
            "type": doc_type
        }
        self.usage_data["history"].append(record)
        
        # Keep history manageable
        if len(self.usage_data["history"]) > 1000:
             self.usage_data["history"] = self.usage_data["history"][-1000:]
             
        self._save_usage()

    def get_stats(self) -> Dict:
        return {
            "total_pages_processed": self.usage_data["total_pages"],
            "daily_usage": self._get_daily_usage()
        }

    def _get_daily_usage(self) -> int:
        today = datetime.now().date().isoformat()
        return sum(
            r["pages"] for r in self.usage_data["history"] 
            if r["timestamp"].startswith(today)
        )
