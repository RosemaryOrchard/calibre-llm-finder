#!/usr/bin/env bash
# Quick diagnostic for "500 Internal Server Error" from Ollama's embeddings
# endpoint during `clf index`. Run this on the machine running Ollama, then
# check diagnostics.log in the project root (or sync it up via Dropbox).
set -uo pipefail

OUT="$(dirname "$0")/../diagnostics.log"

{
  echo "=== $(date) ==="
  echo
  echo "--- ollama list ---"
  ollama list

  echo
  echo "--- ollama pull nomic-embed-text ---"
  ollama pull nomic-embed-text

  echo
  echo "--- curl /api/embeddings ---"
  curl -s -w "\nHTTP_STATUS:%{http_code}\n" \
    http://localhost:11434/api/embeddings \
    -d '{"model":"nomic-embed-text","prompt":"test"}'

  echo
  echo "--- ollama ps (loaded models) ---"
  ollama ps
} > "$OUT" 2>&1

echo "Wrote diagnostics to $OUT"
