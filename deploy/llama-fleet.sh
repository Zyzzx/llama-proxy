#!/usr/bin/env bash
#
# llama-fleet.sh — start a fleet of llama-server backends (one per GPU) under
# `screen`, checking GPU availability first and starting them ONE BY ONE with a
# delay between each.
#
#   * Single file. No config file, no dependencies beyond bash + screen.
#   * Runs on the GPU box (yl6800). It does NOT manage llm-proxy.
#   * Monitoring + shutdown are done by hand with `screen` (see below).
#
# Usage:
#   ./llama-fleet.sh start          start every backend (GPUS of them), one by one
#   ./llama-fleet.sh start 0 2 4    start only backends on GPUs 0, 2, 4
#   ./llama-fleet.sh check          run the GPU availability preflight only
#   ./llama-fleet.sh list           show the GPU -> port -> alias -> screen map
#   ./llama-fleet.sh help           this text
#
# Operate by hand afterwards (logs live in each screen — attach to see them):
#   screen -ls                          # see the fleet: 9B-0 .. 9B-4
#   screen -r 9B-0                      # attach + watch logs (Ctrl-a then d)
#   screen -S 9B-0 -X quit              # stop one (SIGTERM; screen cleans up)
#
# The fleet is index-driven: for GPU i in 0..GPUS-1,
#   HIP_VISIBLE_DEVICES = i , --port = PORT_BASE+i , -a = ${ALIAS_BASE}-i ,
#   screen session      = ${ALIAS_BASE}-i
# Everything else is shared and passed verbatim (see BASE_ARGS below).
# Every value in the CONFIG block can be overridden from the environment, e.g.
#   START_DELAY=10 READY_TIMEOUT=300 ./llama-fleet.sh start

set -u -o pipefail
export LC_ALL=C

# ================================ CONFIG ====================================
# Edit these for your box. (Or override any via the environment.)
LLAMA_BIN="${LLAMA_BIN:-/home/yl/lcrocm/build/bin/llama-server}"
MODEL="${MODEL:-/home/yl/llms/Ornith-1.5-9B-Q8_0.gguf}"
MMPROJ="${MMPROJ:-/home/yl/llms/mmproj-Ornith-1.5-9B-F16.gguf}"
CHAT_TPL="${CHAT_TPL:-/home/yl/llms/Qwentempfix.jinja}"

GPUS="${GPUS:-5}"                  # how many backends/GPUs
PORT_BASE="${PORT_BASE:-8080}"     # backend i listens on PORT_BASE+i
ALIAS_BASE="${ALIAS_BASE:-9B}"     # backend i alias + screen name: 9B-i
HOST="${HOST:-0.0.0.0}"            # --host for every backend

START_DELAY="${START_DELAY:-20}"   # seconds between starting consecutive backends
READY_TIMEOUT="${READY_TIMEOUT:-180}"  # seconds to wait for a backend to serve
READY_POLL="${READY_POLL:-2}"      # seconds between readiness polls

# Optional extra flags appended per backend (leave empty to use none).
# Example: EXTRA_ARGS="--api-key sk-local"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# ============================================================================

# The shared llama-server flags, verbatim (index/alias/port added per GPU).
BASE_ARGS="-m $MODEL --mmproj $MMPROJ -ngl 99 --no-mmap -np 1 --flash-attn on --ctx-size 160000 -jinja --repeat-penalty 1.0 --presence-penalty 0.0 -ts 15 --log-prefix --log-timestamps -s 34632165467897 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.00 -rea on --reasoning-budget 1024 --chat-template-file $CHAT_TPL --metrics"

# --- helpers ----------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 1; }

screen_name()  { printf '%s-%s' "$ALIAS_BASE" "$1"; }   # 9B-0
port_of()      { printf '%s' "$((PORT_BASE + $1))"; }   # 8080
alias_of()     { printf '%s-%s' "$ALIAS_BASE" "$1"; }   # 9B-0

# --- tools (locate once, degrade gracefully) --------------------------------
# find_tool NAME [extra paths...] -> prints the first resolvable path, or ""
find_tool() {
    local n="$1"; shift
    if command -v "$n" >/dev/null 2>&1; then
        command -v "$n"; return
    fi
    local p
    for p in "$@"; do
        [ -x "$p" ] && { printf '%s' "$p"; return; }
    done
}
ROCM_SMI="$(find_tool rocm-smi /opt/rocm/bin/rocm-smi)"
SS_CMD="$(find_tool ss)"
CURL="$(find_tool curl)"
have() { command -v "$1" >/dev/null 2>&1; }

# --- per-GPU facts ----------------------------------------------------------
# Sets GPU_STATUS = present | missing | unknown (version-agnostic).
# "missing" only when a reliable detector actually reports absence; "unknown"
# when no detector is available (so we never block a start on a detection gap).
check_gpu_present() {
    local i="$1" checked=0
    # KFD topology node — most reliable for AMD compute GPUs (node id == GPU index)
    [ -d /sys/class/kfd/kfd/topology/nodes ] && checked=1
    [ -e "/sys/class/kfd/kfd/topology/nodes/$i" ] && { GPU_STATUS=present; return; }
    # DRM card node
    [ -d /sys/class/drm ] && checked=1
    [ -e "/sys/class/drm/card$i" ] && { GPU_STATUS=present; return; }
    # rocm-smi --showid (last resort)
    if [ -n "$ROCM_SMI" ]; then
        checked=1
        "$ROCM_SMI" --showid 2>/dev/null | grep -Eqi "(^|[^0-9])${i}([^0-9]|$)" && { GPU_STATUS=present; return; }
    fi
    if [ "$checked" -eq 1 ]; then GPU_STATUS=missing; else GPU_STATUS=unknown; fi
}

# VRAM used by GPU index, in MiB (rocm-smi). Empty if unknown.
gpu_vram_used() {
    local i="$1" out
    [ -n "$ROCM_SMI" ] || return 0
    out="$("$ROCM_SMI" 2>/dev/null)"
    # two layouts, both have a single "GPU[N]" header line per card.
    # (kept single-line so the script survives copy/paste intact)
    if printf '%s\n' "$out" | grep -Eq '^\s*GPU\[[0-9]+\]\s*Card'; then
        # new layout:  | GPU[0] : Card 0: ... | GPU use (%) | Memory use (MiB) |
        printf '%s\n' "$out" | awk -v n="$i" '$0 ~ "GPU\\[" n "\\]" { card=1 } card && /Memory use/ { for (f=NF; f>=1; f--) if ($f ~ /^[0-9]+$/) { print $f; exit } }'
    else
        # older layout:  GPU[0] : ...  | 100%  |  8 MiB  /  16384 MiB
        printf '%s\n' "$out" | awk -v n="$i" '$0 ~ "GPU\\[" n "\\]" { card=1 } card { for (f=NF; f>=1; f--) if ($f ~ /^MiB$/) { print $(f-1); exit } }'
    fi
}

# Is the port being served by a llama-server? ("yes"/"no"; "" if unknown)
port_serving() {
    local port="$1"
    if [ -n "$CURL" ]; then
        if "$CURL" -fsS -m 2 -o /dev/null "http://127.0.0.1:${port}/metrics" 2>/dev/null; then
            printf 'yes'; return
        fi
        if "$CURL" -fsS -m 2 -o /dev/null "http://127.0.0.1:${port}/health" 2>/dev/null; then
            printf 'yes'; return
        fi
        printf 'no'; return
    fi
    # fall back to a socket listener check
    if [ -n "$SS_CMD" ]; then
        if "$SS_CMD" -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}\$"; then
            printf 'yes'
        else
            printf 'no'
        fi
    fi
}

# Is the screen session alive?
# `screen -ls` lists a session as "PID.name" (e.g. "73342.9B-1"), so match the
# name as a dot-prefixed token — this works for detached and running alike.
screen_alive() {
    screen -ls 2>/dev/null | grep -Eq "\\.${1}([[:space:]]|\$)"
}

# Wait until the port serves (or timeout). Prints the outcome word.
wait_ready() {
    local port="$1" deadline=$(( $(date +%s) + READY_TIMEOUT ))
    while :; do
        [ "$(port_serving "$port")" = "yes" ] && { printf 'ready'; return; }
        if screen_alive "$(screen_name "$2")"; then
            [ "$(date +%s)" -ge "$deadline" ] && { printf 'timeout'; return; }
            sleep "$READY_POLL"
        else
            printf 'died'; return   # screen gone before it ever served
        fi
    done
}

# --- preflight: check availability -----------------------------------------
check_gpus() {
    local i vram port srv status
    log "checking GPU availability (GPUS=$GPUS) ..."
    [ -n "$ROCM_SMI" ] || warn "rocm-smi not found; VRAM 'free?' hint disabled (presence check still works)"

    # required files + tools (hard requirements)
    have screen || die "screen not found (apt install screen)"
    [ -x "$LLAMA_BIN" ] || die "llama-server binary missing/not executable: $LLAMA_BIN"
    [ -f "$MODEL"    ] || die "model missing: $MODEL"
    [ -f "$MMPROJ"   ] || die "mmproj missing: $MMPROJ"
    [ -f "$CHAT_TPL" ] || die "chat template missing: $CHAT_TPL"

    # per-GPU status (presence is version-agnostic; VRAM is best-effort)
    local missing=0 unknown=0
    for i in $(seq 0 $((GPUS-1))); do
        check_gpu_present "$i"
        port="$(port_of "$i")"
        srv="$(port_serving "$port")"
        vram="$(gpu_vram_used "$i")"
        status="ok"
        case "$GPU_STATUS" in
            missing) status="MISSING"; missing=$((missing+1)) ;;
            unknown) status="unknown"; unknown=$((unknown+1)) ;;
        esac
        if [ "$status" = "ok" ] && [ "$srv" = "yes" ]; then
            status="already running"
        elif [ "$status" = "ok" ] && [ -n "$vram" ] && [ "$vram" -gt 2048 ]; then
            warn "GPU $i (port $port) already uses ${vram} MiB VRAM — something is running there"
            status="busy"
        fi
        log "  GPU $i  port $port  present=${GPU_STATUS}  vram=${vram:-?} MiB  state=$status"
    done

    [ "$unknown" -gt 0 ] && warn "could not confirm presence of $unknown GPU(s) — proceeding"
    [ "$missing" -gt 0 ] && warn "$missing GPU(s) MISSING — those will be skipped on start"
    return 0
}

# --- start one backend ------------------------------------------------------
start_one() {
    local i="$1" name port cmd
    name="$(screen_name "$i")"
    port="$(port_of "$i")"

    check_gpu_present "$i"
    if [ "$GPU_STATUS" = "missing" ]; then
        warn "GPU $i  MISSING — skipping (no such GPU visible)"
        return 0
    fi

    if screen_alive "$name"; then
        log "GPU $i  $name  already running (screen alive) — skipping"
        return 0
    fi
    if [ "$(port_serving "$port")" = "yes" ]; then
        log "GPU $i  port $port  already serving (no screen) — skipping"
        return 0
    fi

    # env sets the GPU for the child; logs go to the screen (view them with
    # `screen -r $name`). When llama-server exits, the wrapper shell — and
    # thus the screen session — ends with it.
    cmd="env HIP_VISIBLE_DEVICES=$i $LLAMA_BIN $BASE_ARGS --host $HOST --port $port -a $(alias_of "$i") $EXTRA_ARGS"
    log "GPU $i  starting $name  (port $port, alias $(alias_of "$i")) ..."

    if ! screen -dmS "$name" bash -lc 'source ~/.bashrc 2>/dev/null; source ~/.profile 2>/dev/null; source ~/.bash_profile 2>/dev/null; '"$cmd"' </dev/null'; then
        warn "GPU $i  failed to create screen $name"
        return 1
    fi

    local outcome
    outcome="$(wait_ready "$port" "$i")"
    case "$outcome" in
        ready)   log "GPU $i  $name  READY  (port $port)  — view: screen -r $name" ;;
        timeout) warn "GPU $i  $name  still loading after ${READY_TIMEOUT}s — view: screen -r $name" ;;
        died)    warn "GPU $i  $name  exited early — view: screen -r $name (or check it did not OOM)" ;;
    esac
    return 0
}

# --- commands ---------------------------------------------------------------
cmd_list() {
    printf '%-4s %-8s %-6s %-10s\n' "GPU" "PORT" "ALIAS" "SCREEN"
    local i
    for i in $(seq 0 $((GPUS-1))); do
        printf '%-4s %-8s %-6s %-10s\n' \
            "$i" "$(port_of "$i")" "$(alias_of "$i")" "$(screen_name "$i")"
    done
}

cmd_check() {
    check_gpus
    log "preflight done"
}

cmd_start() {
    local only_all=1
    [ "$#" -gt 0 ] && only_all=0

    check_gpus

    local i
    if [ "$only_all" -eq 1 ]; then
        for i in $(seq 0 $((GPUS-1))); do
            start_one "$i"
            [ "$i" -lt $((GPUS-1)) ] && { log "waiting ${START_DELAY}s before next backend ..."; sleep "$START_DELAY"; }
        done
    else
        local prev=-1
        for i in "$@"; do
            case "$i" in (*[!0-9]*|'') die "invalid GPU index: $i (expected an integer 0..$((GPUS-1)))" ;; esac
            [ "$i" -ge "$GPUS" ] && { warn "GPU $i out of range (0..$((GPUS-1))) — skipping"; continue; }
            [ "$prev" -ne -1 ] && { log "waiting ${START_DELAY}s before next backend ..."; sleep "$START_DELAY"; }
            start_one "$i"
            prev="$i"
        done
    fi

    log "done. screens:"
    screen -ls 2>/dev/null | grep -E "$ALIAS_BASE-[0-9]+" || true
}

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
    local cmd="${1:-help}"
    [ "$#" -gt 0 ] && shift
    case "$cmd" in
        start)   cmd_start "$@" ;;
        check)   cmd_check ;;
        list)    cmd_list ;;
        help|-h|--help) usage ;;
        *) echo "unknown command: $cmd"; usage; exit 2 ;;
    esac
}

main "$@"
