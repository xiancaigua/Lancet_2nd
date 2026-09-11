#!/usr/bin/env python3
"""Send an explicitly requested Lancet SMTP notification from the Host."""

from __future__ import annotations

import argparse
import datetime as dt
import smtplib
import socket
import ssl
from email.message import EmailMessage
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = REPO / "configs/local/lancet_email.env"


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _bool(values: dict[str, str], key: str) -> bool:
    value = values.get(key, "").lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{key} must be true or false")
    return value == "true"


def _validated(values: dict[str, str], *, explicit_test: bool) -> dict[str, object]:
    required = (
        "LANCET_SMTP_HOST",
        "LANCET_SMTP_PORT",
        "LANCET_SMTP_USER",
        "LANCET_SMTP_PASSWORD",
        "LANCET_EMAIL_FROM",
        "LANCET_EMAIL_TO",
        "LANCET_SMTP_STARTTLS",
        "LANCET_SMTP_SSL",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    enabled = values.get("LANCET_EMAIL_ENABLED", "false").lower() == "true"
    if not enabled and not explicit_test:
        raise ValueError("Email is disabled; use --test only for an explicit test send")
    starttls = _bool(values, "LANCET_SMTP_STARTTLS")
    direct_ssl = _bool(values, "LANCET_SMTP_SSL")
    if starttls == direct_ssl:
        raise ValueError("Enable exactly one of LANCET_SMTP_STARTTLS and LANCET_SMTP_SSL")
    port = int(values["LANCET_SMTP_PORT"])
    if not 1 <= port <= 65535:
        raise ValueError("LANCET_SMTP_PORT must be between 1 and 65535")
    return {**values, "port": port, "starttls": starttls, "ssl": direct_ssl}


def _send(config: dict[str, object], message: EmailMessage) -> None:
    context = ssl.create_default_context()
    host = str(config["LANCET_SMTP_HOST"])
    port = int(config["port"])
    if bool(config["ssl"]):
        client_context = smtplib.SMTP_SSL(host, port, timeout=20, context=context)
    else:
        client_context = smtplib.SMTP(host, port, timeout=20)
    with client_context as client:
        if bool(config["starttls"]):
            client.ehlo()
            client.starttls(context=context)
            client.ehlo()
        client.login(
            str(config["LANCET_SMTP_USER"]),
            str(config["LANCET_SMTP_PASSWORD"]),
        )
        client.send_message(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Explicitly authorize one test message even when ENABLED=false.",
    )
    args = parser.parse_args()
    if not args.test:
        raise SystemExit("Only explicit test sends are currently supported; pass --test")

    try:
        config = _validated(_load_env(args.env_file), explicit_test=True)
        now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        message = EmailMessage()
        message["Subject"] = f"[Lancet] SMTP test {now}"
        message["From"] = str(config["LANCET_EMAIL_FROM"])
        message["To"] = str(config["LANCET_EMAIL_TO"])
        message.set_content(
            "Lancet SMTP configuration test succeeded.\n\n"
            f"Host: {socket.gethostname()}\n"
            f"Time: {now}\n"
            "This message is not connected to the training lifecycle.\n"
        )
        _send(config, message)
    except smtplib.SMTPAuthenticationError as exc:
        print(f"SMTP authentication failed (code {exc.smtp_code}).")
        return 2
    except (OSError, smtplib.SMTPException, ValueError) as exc:
        print(f"SMTP test failed: {type(exc).__name__}: {exc}")
        return 2

    print("SMTP test sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
