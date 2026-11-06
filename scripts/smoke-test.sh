#!/usr/bin/env bash
# End-to-end check against a running server.
#   kubectl -n tts port-forward svc/qwen3-tts 8005:8005 &
#   ./scripts/smoke-test.sh my-clip.wav "Exactly what is said in the clip."
set -euo pipefail

BASE="${BASE:-http://localhost:8005}"
CLIP="${1:?usage: $0 <reference.wav|mp3|m4a> \"<transcript of the clip>\"}"
TRANSCRIPT="${2:-}"
NAME="${NAME:-smoketest}"

echo "== health"
until curl -fsS "$BASE/healthz"; do echo " (model still loading, retrying in 10s)"; sleep 10; done
echo

echo "== save voice '$NAME'"
curl -fsS -X POST "$BASE/v1/voices" \
  -F "name=$NAME" -F "file=@$CLIP" -F "ref_text=$TRANSCRIPT" -F "language=English" -F "overwrite=true"
echo

echo "== list voices"
curl -fsS "$BASE/v1/audio/voices"
echo

echo "== OpenAI-style speech -> out.mp3"
time curl -fsS -X POST "$BASE/v1/audio/speech" -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3-tts\",\"voice\":\"$NAME\",\"input\":\"This is a test of my cloned voice running on the cluster.\",\"response_format\":\"mp3\",\"seed\":42}" \
  -o out.mp3
ls -l out.mp3

echo "== one-shot clone (nothing saved) -> oneshot.wav"
curl -fsS -X POST "$BASE/v1/audio/clone" \
  -F "file=@$CLIP" -F "ref_text=$TRANSCRIPT" -F "text=One-shot cloning works too." -o oneshot.wav
ls -l oneshot.wav

echo "OK. Play out.mp3 and oneshot.wav."
