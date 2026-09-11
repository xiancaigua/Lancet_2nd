"""Safety-focused tests for the opt-in Lancet SMTP sender."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/experiments/notify_email.py"
SPEC = importlib.util.spec_from_file_location("lancet_notify_email", SCRIPT)
notify = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(notify)


def _values() -> dict[str, str]:
    return {
        "LANCET_EMAIL_ENABLED": "false",
        "LANCET_SMTP_HOST": "smtp.example.com",
        "LANCET_SMTP_PORT": "587",
        "LANCET_SMTP_USER": "sender@example.com",
        "LANCET_SMTP_PASSWORD": "app-password",
        "LANCET_EMAIL_FROM": "sender@example.com",
        "LANCET_EMAIL_TO": "receiver@example.com",
        "LANCET_SMTP_STARTTLS": "true",
        "LANCET_SMTP_SSL": "false",
    }


def test_disabled_config_requires_explicit_test():
    with pytest.raises(ValueError, match="Email is disabled"):
        notify._validated(_values(), explicit_test=False)
    assert notify._validated(_values(), explicit_test=True)["port"] == 587


def test_exactly_one_tls_mode_is_required():
    values = _values()
    values["LANCET_SMTP_SSL"] = "true"
    with pytest.raises(ValueError, match="exactly one"):
        notify._validated(values, explicit_test=True)


def test_env_loader_preserves_password_characters(tmp_path):
    env_file = tmp_path / "mail.env"
    env_file.write_text(
        "LANCET_SMTP_PASSWORD='a#b=c'\nLANCET_SMTP_PORT=587\n",
        encoding="utf-8",
    )
    values = notify._load_env(env_file)
    assert values["LANCET_SMTP_PASSWORD"] == "a#b=c"
