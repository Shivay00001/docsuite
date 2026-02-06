import hashlib
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class LicenseManager:
    """
    Manages license validation and feature gating.
    Supports offline validation via signed keys (simulated for now).
    """

    def __init__(self, license_key: Optional[str] = None, public_key: Optional[str] = None):
        self.license_key = license_key
        self.public_key = public_key
        self._cached_plan = "free"
        self._last_validated = None

    def validate(self) -> Dict[str, Any]:
        """
        Validates the license key.
        Returns license details dict if valid, or raises exception/returns defaults.
        """
        if not self.license_key:
            return {"plan": "free", "valid": True, "features": ["ocr_basic"]}

        # Simulate validation logic (In prod, verify signature with public_key)
        try:
            # Mock structure: "PLAN-EXPIRY-SIGNATURE"
            parts = self.license_key.split("-")
            if len(parts) < 2:
                raise ValueError("Invalid key format")
            
            plan = parts[0].lower()
            expiry_str = parts[1]
            
            # Check expiry
            # expiry is YYYYMMDD
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d")
            if datetime.now() > expiry_date:
                 return {"plan": "free", "valid": False, "reason": "expired", "features": ["ocr_basic"]}

            self._cached_plan = plan
            return {
                "plan": plan,
                "valid": True,
                "expires": expiry_str,
                "features": self._get_features_for_plan(plan)
            }

        except Exception as e:
            return {"plan": "free", "valid": False, "reason": str(e), "features": ["ocr_basic"]}

    def _get_features_for_plan(self, plan: str) -> list:
        features = {
            "free": ["ocr_basic"],
            "pro": ["ocr_basic", "ocr_advanced", "batch_processing", "priority_support"],
            "enterprise": ["ocr_basic", "ocr_advanced", "batch_processing", "priority_support", "on_prem_deployment", "custom_models"]
        }
        return features.get(plan, ["ocr_basic"])

    def check_feature(self, feature_name: str) -> bool:
        """Check if a specific feature is allowed by the current license."""
        details = self.validate()
        return feature_name in details.get("features", [])
