#!/usr/bin/env bash
# Fetch the authors' official code at the exact commit used for the replication.
# The code is MIT-licensed and kept out of this repo (replication/upstream is git-ignored).
set -euo pipefail

REPO="https://github.com/mjs19971997/ensemble-dqn-5g-handover.git"
COMMIT="b61e399"
DEST="$(dirname "$0")/upstream"

if [ ! -d "$DEST/.git" ]; then
    git clone --quiet "$REPO" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "Authors' code ready in $DEST at commit $COMMIT"
