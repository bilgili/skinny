#!/usr/bin/env bash
# guarded_metal.sh — run a Metal-touching command under memory + timeout guards.
#
# WHY THIS EXISTS
#   Cold-compiling the full main_pass.slang megakernel for the `metal` target
#   spikes RAM inside Apple's MTLCompilerService XPC daemon (a SEPARATE process
#   from ours — we cannot ulimit it; macOS arm64 ignores RLIMIT_AS anyway). This
#   machine runs with vm.swapusage total = 0, so an overcommit has nowhere to go
#   and the box locks / reboots. Worse: SIGKILLing a mid-compile Metal process
#   wedges MTLCompilerService and can force a restart.
#
# WHAT THIS DOES
#   * Preflight: refuses to start unless enough system memory is free and no
#     MTLCompilerService looks wedged.
#   * Watchdog: polls SYSTEM-WIDE free memory (the real signal — the spike is in
#     MTLCompilerService, not our child's RSS) plus child RSS and a wall-clock
#     timeout, every second.
#   * On any breach it stops the child GRACEFULLY ONLY: SIGINT, wait, SIGTERM,
#     wait. It NEVER sends SIGKILL (that is what wedges the compiler service).
#
# USAGE
#   scripts/guarded_metal.sh -- <command> [args...]
#   MIN_FREE_GB=10 FLOOR_FREE_GB=8 TIMEOUT_S=300 scripts/guarded_metal.sh -- python probe.py
#
# TUNABLES (env)
#   MIN_FREE_GB   preflight floor; abort if less is free at start   (default 10)
#   FLOOR_FREE_GB watchdog floor; stop child if free drops below    (default 8)
#   RSS_CAP_GB    watchdog cap; stop child if its RSS exceeds        (default 18)
#   TIMEOUT_S     watchdog wall-clock; stop child after this long    (default 300)
#   GRACE_S       seconds to wait after each graceful signal         (default 25)
#   POLL_S        watchdog poll interval                             (default 1)
#   MTLC_WEDGE_GB preflight: abort if MTLCompilerService total RSS   (default 4)
#                 already exceeds this (likely wedged from a crash)
set -u

MIN_FREE_GB=${MIN_FREE_GB:-10}
FLOOR_FREE_GB=${FLOOR_FREE_GB:-8}
RSS_CAP_GB=${RSS_CAP_GB:-18}
TIMEOUT_S=${TIMEOUT_S:-300}
GRACE_S=${GRACE_S:-25}
POLL_S=${POLL_S:-1}
MTLC_WEDGE_GB=${MTLC_WEDGE_GB:-4}
PAGE=16384

log() { printf '[guard %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

avail_gb() {
  # free + speculative + purgeable pages -> GB (conservative "truly free")
  vm_stat | awk -v pg="$PAGE" '
    /Pages free/        {gsub(/\./,"",$3); f=$3}
    /Pages speculative/ {gsub(/\./,"",$3); s=$3}
    /Pages purgeable/   {gsub(/\./,"",$3); p=$3}
    END { printf "%.2f", (f+s+p)*pg/1073741824 }'
}
mtlc_rss_gb() {
  local pids; pids=$(pgrep -f MTLCompilerService.xpc | tr '\n' ' ')
  [ -z "$pids" ] && { echo "0.00"; return; }
  ps -o rss= $pids 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1048576}'
}
swap_used_mb() { sysctl -n vm.swapusage 2>/dev/null | awk '{gsub(/M/,"",$6); print $6+0}'; }
lt() { awk "BEGIN{exit !($1 < $2)}"; }   # $1 < $2 ?
gt() { awk "BEGIN{exit !($1 > $2)}"; }   # $1 > $2 ?

# ---- arg parsing: everything after `--` is the command ----
[ "${1:-}" = "--" ] && shift
[ $# -eq 0 ] && { log "usage: guarded_metal.sh -- <command> [args...]"; exit 2; }

# ---- preflight ----
A=$(avail_gb); M=$(mtlc_rss_gb)
log "preflight: avail=${A}GB (need>=${MIN_FREE_GB}) | MTLCompilerService total=${M}GB (wedge>${MTLC_WEDGE_GB}) | swap_used=$(swap_used_mb)MB"
if lt "$A" "$MIN_FREE_GB"; then
  log "ABORT: only ${A}GB free (< ${MIN_FREE_GB}GB). Close apps or wait; not starting a Metal compile."
  exit 3
fi
if gt "$M" "$MTLC_WEDGE_GB"; then
  log "ABORT: MTLCompilerService total RSS ${M}GB looks wedged (> ${MTLC_WEDGE_GB}GB). Reboot before retrying."
  exit 4
fi

# macOS SIP strips DYLD_* from the environment when this protected `/bin/bash`
# was exec'd, but leaves VULKAN_SDK intact. `renderer.py` imports `vulkan`
# unconditionally (even the Metal path), and that binding loads libvulkan off the
# dylib path — so re-derive DYLD_LIBRARY_PATH from the surviving VULKAN_SDK before
# launching the (non-protected) child. No-op when the SDK isn't set.
if [ -n "${VULKAN_SDK:-}" ] && [ -z "${DYLD_LIBRARY_PATH:-}" ]; then
  export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib"
  log "re-derived DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH (SIP stripped it from /bin/bash)"
fi

# ---- launch ----
log "launch: $*"
"$@" &
CHILD=$!
log "child pid=$CHILD  caps: floor=${FLOOR_FREE_GB}GB rss=${RSS_CAP_GB}GB timeout=${TIMEOUT_S}s poll=${POLL_S}s"

stop_child() {   # graceful only. SIGINT -> wait -> SIGTERM -> wait. NEVER -9.
  local why="$1" i
  log "STOP ($why) -> SIGINT pid $CHILD (lets it unwind & release the Metal device cleanly)"
  kill -INT "$CHILD" 2>/dev/null
  for ((i=0;i<GRACE_S;i++)); do kill -0 "$CHILD" 2>/dev/null || { log "child exited after SIGINT"; return 0; }; sleep 1; done
  log "alive after ${GRACE_S}s -> SIGTERM pid $CHILD"
  kill -TERM "$CHILD" 2>/dev/null
  for ((i=0;i<GRACE_S;i++)); do kill -0 "$CHILD" 2>/dev/null || { log "child exited after SIGTERM"; return 0; }; sleep 1; done
  log "WARNING: pid $CHILD STILL ALIVE. Refusing SIGKILL (would wedge MTLCompilerService -> possible reboot)."
  log "WARNING: leave it; it should finish its current compile and then honor the signal. Do NOT 'kill -9'."
  return 1
}

# graceful on Ctrl-C of the guard itself
trap 'log "guard caught SIGINT"; stop_child "guard interrupted"; exit 130' INT TERM

PEAK_MTLC=0; LOW_AVAIL=$A; start=$SECONDS
while kill -0 "$CHILD" 2>/dev/null; do
  A=$(avail_gb); M=$(mtlc_rss_gb)
  RSS_KB=$(ps -o rss= -p "$CHILD" 2>/dev/null | tr -d ' '); RSS_KB=${RSS_KB:-0}
  RSS_GB=$(awk "BEGIN{printf \"%.2f\", $RSS_KB/1048576}")
  EL=$((SECONDS - start))
  gt "$M" "$PEAK_MTLC" && PEAK_MTLC=$M
  lt "$A" "$LOW_AVAIL" && LOW_AVAIL=$A
  log "t=${EL}s avail=${A}GB child_rss=${RSS_GB}GB mtlc=${M}GB"
  if lt "$A" "$FLOOR_FREE_GB"; then stop_child "free ${A}GB < ${FLOOR_FREE_GB}GB"; break; fi
  if gt "$RSS_GB" "$RSS_CAP_GB"; then stop_child "child rss ${RSS_GB}GB > ${RSS_CAP_GB}GB"; break; fi
  if [ "$EL" -ge "$TIMEOUT_S" ]; then stop_child "timeout ${EL}s >= ${TIMEOUT_S}s"; break; fi
  sleep "$POLL_S"
done

wait "$CHILD" 2>/dev/null; rc=$?
log "DONE rc=$rc | low_avail=${LOW_AVAIL}GB | peak_MTLCompilerService=${PEAK_MTLC}GB | final_avail=$(avail_gb)GB swap=$(swap_used_mb)MB"
exit "$rc"
