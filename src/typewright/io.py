from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import DataError


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_document(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > 8_000_000:
        raise DataError(f"Document exceeds the 8 MB safety limit: {path.name}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        # SafeLoader does not execute Python objects. Duplicate keys are rejected.
        class UniqueLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise DataError("YAML keys must be strings; quote 'true' and 'false' in Noul criteria.")
                if key in result:
                    raise DataError(f"Duplicate YAML key: {key}")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        try:
            data = yaml.load(text, Loader=UniqueLoader)
        except yaml.YAMLError as exc:
            raise DataError("Invalid or unsupported YAML; no input values are included in this error.") from exc
    else:
        data = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    if not isinstance(data, dict):
        raise DataError("The document root must be an object.")
    canonical(data)  # Reject non-JSON YAML values and non-finite numbers.
    return data


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise DataError(f"Non-finite JSON value: {value}")


def json_loads(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".s1-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
