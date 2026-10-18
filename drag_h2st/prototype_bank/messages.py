"""Stable message keys with English API fallbacks; the UI selects translations."""
import json
from pathlib import Path

ENGLISH = json.loads((Path(__file__).parent / "static/translations.json").read_text(encoding="utf-8"))["en"]


def message(key, **params):
    return {"message": ENGLISH[key].format(**params), "message_key": key, "message_params": params}


class UserError(ValueError):
    def __init__(self, key, status=400, **params):
        self.key, self.params, self.status = key, params, status
        super().__init__(ENGLISH[key].format(**params))


def error_payload(error):
    if isinstance(error, UserError):
        key, params = error.key, error.params
    elif isinstance(error, FileNotFoundError):
        key, params = "error.fileMissing", {}
    elif isinstance(error, (KeyError, ValueError)):
        key, params = "error.invalidRequest", {}
    else:
        key, params = "error.serverFailure", {}
    return {"error": ENGLISH[key].format(**params), "error_key": key, "error_params": params}
