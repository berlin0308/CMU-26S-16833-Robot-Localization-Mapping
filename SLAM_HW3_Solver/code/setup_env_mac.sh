#!/usr/bin/env bash
# macOS: use Homebrew SuiteSparse when building sparseqr (avoids Anaconda include paths).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SP="$(brew --prefix suite-sparse 2>/dev/null || true)"
if [[ -z "$SP" || ! -d "$SP/include/suitesparse" ]]; then
  echo "Install SuiteSparse first: brew install suite-sparse" >&2
  exit 1
fi
export PKG_CONFIG_PATH="${SP}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CPPFLAGS="-I${SP}/include/suitesparse ${CPPFLAGS:-}"
export LDFLAGS="-L${SP}/lib ${LDFLAGS:-}"
cd "$ROOT"
uv sync "$@"
