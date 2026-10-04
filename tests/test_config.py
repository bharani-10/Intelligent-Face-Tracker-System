import json

import pytest

from app.config import ConfigurationError, load_config


def test_valid_config_accepts_legacy_similarity_threshold(tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "input": {"source_type": "file", "source": "video.mp4"},
        "detection": {"confidence": 0.5, "frame_skip": 2},
        "recognition": {"similarity_threshold": 0.6},
        "tracking": {"max_age": 30},
        "database": {"url_env": "DATABASE_URL"},
        "logging": {"path": "logs/events.log"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert load_config(str(config_path))["recognition"]["threshold"] == 0.6


def test_invalid_frame_skip_is_rejected(tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "input": {"source_type": "file", "source": "video.mp4"},
        "detection": {"confidence": 0.5, "frame_skip": 0},
        "recognition": {"threshold": 0.6},
        "tracking": {"max_age": 30},
        "database": {"url_env": "DATABASE_URL"},
        "logging": {"path": "logs/events.log"},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="frame_skip"):
        load_config(str(config_path))