import json
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


def load_config(path: str = "config.json") -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"Unable to load configuration {path}: {error}") from error

    required_sections = {"input", "detection", "recognition", "tracking", "database", "logging"}
    missing_sections = required_sections - config.keys()
    if missing_sections:
        raise ConfigurationError(f"Missing configuration sections: {sorted(missing_sections)}")

    input_config = config["input"]
    if input_config.get("source_type") not in {"file", "rtsp"} or not input_config.get("source"):
        raise ConfigurationError("input.source_type must be 'file' or 'rtsp', and input.source is required")

    detection = config["detection"]
    _bounded_number(detection.get("confidence"), "detection.confidence")
    frame_skip = detection.get("frame_skip")
    if not isinstance(frame_skip, int) or frame_skip < 1:
        raise ConfigurationError("detection.frame_skip must be an integer >= 1")

    recognition = config["recognition"]
    threshold = recognition.get("threshold", recognition.get("similarity_threshold"))
    _bounded_number(threshold, "recognition.threshold")
    tracking_age = config["tracking"].get("max_age")
    if not isinstance(tracking_age, int) or tracking_age < 1:
        raise ConfigurationError("tracking.max_age must be an integer >= 1")
    database_env = config["database"].get("url_env", "DATABASE_URL")
    if not isinstance(database_env, str) or not database_env:
        raise ConfigurationError("database.url_env must name the PostgreSQL URL environment variable")
    if not isinstance(config["logging"].get("path"), str) or not config["logging"]["path"]:
        raise ConfigurationError("logging.path is required")

    config["recognition"]["threshold"] = float(threshold)
    return config


def _bounded_number(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ConfigurationError(f"{name} must be a number between 0 and 1")