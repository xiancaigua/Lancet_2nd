#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"

probe="$(mktemp "$repo/.audit_mount.XXXXXX")"
trap 'rm -f "$probe"' EXIT
name="$(basename "$probe")"
printf '%s\n' 'host-visible' >"$probe"
test "$(./dev d4rl cat "$name")" = 'host-visible'
./dev d4rl --shell "printf '%s\\n' 'container-visible' > '$name'"
test "$(cat "$probe")" = 'container-visible'
printf '%s\n' 'mount=read-write-shared'
