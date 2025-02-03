#!/usr/bin/env bash
# Tag a release from the current main. Usage: scripts/release.sh vX.Y.Z
set -euo pipefail

version=${1:?usage: scripts/release.sh vX.Y.Z}

if ! grep -q "$version" CHANGELOG.md 2>/dev/null; then
  echo "warning: no $version section found in CHANGELOG.md" >&2
fi

git tag -a "$version" -m "Release $version"
git push origin "$version"
