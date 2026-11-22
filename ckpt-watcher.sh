#!/usr/bin/env bash
# ckpt-watcher.sh — upload training checkpoints to S3 / Cloudflare R2, but only fully written ones.
#
# Why: if you upload from inside the training loop (or sync a directory blindly), a crash or hang
# mid-upload leaves a corrupt checkpoint in storage — and you resume from garbage. Instead the
# training script drops a marker file into checkpoint-N/ only AFTER the checkpoint is completely
# written, and this watcher uploads only directories that carry the marker.
#
# Directory contract (Hugging Face Trainer, Axolotl, Unsloth, LLaMA-Factory all fit):
#   ckpt/
#     checkpoint-100/           <- complete, has .done marker   -> uploaded
#     tmp-checkpoint-150/       <- still being written          -> skipped (tmp-*)
#     checkpoint-150/           <- complete, has .done marker   -> uploaded
#
# Training side, one line after each save:
#   pathlib.Path(f"{ckpt_dir}/checkpoint-{step}/.done").touch()
#
# Usage:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   export S3_URI=s3://my-bucket/my-job          # target prefix
#   export S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com   # for R2; omit for AWS S3
#   ./ckpt-watcher.sh                            # Ctrl-C or SIGTERM triggers one final sync

set -u

WATCH_DIR="${CKPT_DIR:-/workspace/ckpt}"
MARKER="${READY_MARKER:-.done}"
INTERVAL="${SYNC_INTERVAL_S:-120}"
S3_URI="${S3_URI:?set S3_URI, e.g. s3://bucket/job-name}"
ENDPOINT="${S3_ENDPOINT:-}"

log() { echo "[ckpt-watcher $(date -u +%H:%M:%S)] $*"; }

aws_s3() {
  if [ -n "$ENDPOINT" ]; then aws --endpoint-url "$ENDPOINT" "$@"; else aws "$@"; fi
}

upload_ready() {
  shopt -s nullglob
  local dir name stamp
  for dir in "$WATCH_DIR"/checkpoint-*/; do
    dir="${dir%/}"
    name="$(basename "$dir")"
    case "$name" in tmp-*|*-incomplete) continue ;; esac   # HF Trainer's in-progress dirs
    [ -f "$dir/$MARKER" ] || continue                       # not fully written yet
    stamp="$WATCH_DIR/.uploaded-$name"
    if [ -f "$stamp" ] && [ ! "$dir/$MARKER" -nt "$stamp" ]; then
      continue                                              # already uploaded this version
    fi
    log "uploading $name -> $S3_URI/$name"
    if aws_s3 s3 sync "$dir" "$S3_URI/$name" --only-show-errors; then
      touch "$stamp"
      log "uploaded $name"
    else
      log "FAILED $name — will retry next round"
    fi
  done
}

trap 'log "termination signal — final sync"; upload_ready; log "bye"; exit 0' TERM INT

log "watching $WATCH_DIR every ${INTERVAL}s -> $S3_URI (marker: $MARKER)"
while true; do
  upload_ready
  sleep "$INTERVAL" &
  wait $!
done
