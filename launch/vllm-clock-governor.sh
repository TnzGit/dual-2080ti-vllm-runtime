#!/usr/bin/env bash
set -uo pipefail

METRICS_URL="http://127.0.0.1:8000/metrics"
GPU_INDEX=0
LOCK_RANGE="1800,1800"
POWER_LIMIT_WATTS=200
POLL_SECONDS=1
IDLE_GRACE_SECONDS=120
LOG_FILE="${GOVERNOR_LOG:-/tmp/vllm-clock-governor.log}"

mkdir -p "$(dirname "$LOG_FILE")"
locked=0
idle_seconds=0

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_FILE"
}

apply_power_limits() {
  local gpu
  for gpu in 0 1; do
    if sudo -n /usr/bin/nvidia-smi -i "$gpu" -pl "$POWER_LIMIT_WATTS" >/dev/null; then
      log "set GPU${gpu} power limit to ${POWER_LIMIT_WATTS} W"
    else
      log "ERROR unable to set GPU${gpu} power limit to ${POWER_LIMIT_WATTS} W"
      return 1
    fi
  done
}

lock_gpu() {
  if sudo -n /usr/bin/nvidia-smi -i "$GPU_INDEX" -lgc "$LOCK_RANGE" >/dev/null; then
    locked=1
    log "locked GPU${GPU_INDEX} graphics clock at ${LOCK_RANGE} MHz (request active)"
  else
    log "ERROR unable to lock GPU${GPU_INDEX}; sudo rule or driver command failed"
  fi
}

release_gpu() {
  if sudo -n /usr/bin/nvidia-smi -i "$GPU_INDEX" -rgc >/dev/null; then
    locked=0
    log "restored GPU${GPU_INDEX} dynamic clocks (idle grace elapsed)"
  else
    log "ERROR unable to restore GPU${GPU_INDEX} dynamic clocks"
  fi
}

running_requests() {
  curl -fsS --max-time 2 "$METRICS_URL" 2>/dev/null | awk '
    /^vllm:num_requests_running\{/ { if ($NF + 0 > 0) active = 1 }
    END { print active + 0 }
  '
}

trap 'if [ "$locked" -eq 1 ]; then release_gpu; fi; exit 0' INT TERM EXIT

apply_power_limits || true
log "governor started: power_limit=${POWER_LIMIT_WATTS}W poll=${POLL_SECONDS}s idle_grace=${IDLE_GRACE_SECONDS}s"
while true; do
  running=$(running_requests || printf 0)
  if [ "$running" = "1" ]; then
    idle_seconds=0
    if [ "$locked" -eq 0 ]; then
      lock_gpu
    fi
  elif [ "$locked" -eq 1 ]; then
    idle_seconds=$((idle_seconds + POLL_SECONDS))
    if [ "$idle_seconds" -ge "$IDLE_GRACE_SECONDS" ]; then
      release_gpu
      idle_seconds=0
    fi
  fi
  sleep "$POLL_SECONDS"
done
