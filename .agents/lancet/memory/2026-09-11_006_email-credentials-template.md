# Agent Memory: Email credentials template

- Date: 2026-09-11
- Sequence: 006
- Agent: Codex
- Branch: main
- Commit before: bc7ff78385882877ebbf90e9e408682020699b9a
- Commit after: this commit (credentials template)
- Status: completed

## Goal

Prepare a safe local SMTP credential surface without sending mail or integrating training.

## What changed

- Added a tracked secret-free example and field documentation.
- Added an ignored local file with empty values and notifications disabled.
- Added a specific `.gitignore` rule for the real credential file.

## Key files

- `configs/local/lancet_email.env.example`
- `configs/local/lancet_email.env`
- `configs/local/README.md`

## Validation

Git ignore status and absence of filled credential values must be verified.

## Decisions

Default `LANCET_EMAIL_ENABLED=false`; STARTTLS defaults true and SSL false.

## Problems / caveats

No SMTP sender, test email, or lifecycle integration exists by design.

## Next dependency

The user fills the ignored file; a later explicitly authorized task implements and tests sending.

## Resume hint

Never echo the filled file or pass its password through command-line arguments.
