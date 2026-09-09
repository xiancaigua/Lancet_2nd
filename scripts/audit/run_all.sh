#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"

scripts/audit/audit_runtime.sh
scripts/audit/audit_mount.sh
./dev d4rl python scripts/audit/audit_d4rl.py
scripts/audit/audit_lancet.sh
printf '%s\n' 'all_audits=passed'
