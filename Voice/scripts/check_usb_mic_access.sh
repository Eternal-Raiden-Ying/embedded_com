#!/usr/bin/env bash
# Read-only USB/Android ALSA permission diagnostic.  It never changes groups,
# ACLs, ownership, modes, or device state.
set -u

DEVICE="${1:-/dev/snd/pcmC1D0c}"
CURRENT_USER="$(id -un)"
CURRENT_UID="$(id -u)"
CURRENT_GROUPS="$(id -G)"

echo "[VOICE][MIC] device=$DEVICE"
echo "[VOICE][MIC] user=$CURRENT_USER uid=$CURRENT_UID"
echo "[VOICE][MIC] supplementary_gids=$CURRENT_GROUPS"
echo "[VOICE][MIC] groups=$(id -nG)"

if [ ! -e "$DEVICE" ]; then
  echo "[VOICE][MIC][BLOCKED] device node does not exist"
  echo "[VOICE][MIC] Connect the USB microphone and confirm the ALSA pcm node before retrying."
  exit 2
fi

DEVICE_UID="$(stat -c '%u' "$DEVICE")"
DEVICE_GID="$(stat -c '%g' "$DEVICE")"
DEVICE_MODE="$(stat -c '%a' "$DEVICE")"
DEVICE_OWNER="$(stat -c '%U' "$DEVICE")"
DEVICE_GROUP="$(stat -c '%G' "$DEVICE")"
echo "[VOICE][MIC] node_uid=$DEVICE_UID node_gid=$DEVICE_GID mode=$DEVICE_MODE owner=$DEVICE_OWNER group=$DEVICE_GROUP"

case " $CURRENT_GROUPS " in
  *" $DEVICE_GID "*)
    echo "[VOICE][MIC] gid_membership=PASS (current process has GID $DEVICE_GID)"
    ;;
  *)
    echo "[VOICE][MIC][BLOCKED] gid_membership=FAIL (current process lacks GID $DEVICE_GID)"
    GROUP_ENTRY="$(getent group "$DEVICE_GID" 2>/dev/null || true)"
    if [ -n "$GROUP_ENTRY" ]; then
      EXISTING_GROUP_NAME="${GROUP_ENTRY%%:*}"
      echo "[VOICE][MIC] GID $DEVICE_GID already maps to group '$EXISTING_GROUP_NAME'."
      echo "[VOICE][MIC][SUGGEST] sudo usermod -aG $EXISTING_GROUP_NAME $CURRENT_USER"
    else
      echo "[VOICE][MIC] GID $DEVICE_GID has no local group-name mapping."
      echo "[VOICE][MIC][SUGGEST] sudo groupadd --gid $DEVICE_GID android_audio"
      echo "[VOICE][MIC][SUGGEST] sudo usermod -aG android_audio $CURRENT_USER"
    fi
    echo "[VOICE][MIC] Re-login SSH/VS Code after group changes before retrying."
    ;;
esac

if test -r "$DEVICE"; then
  echo "[VOICE][MIC] readable=PASS"
else
  echo "[VOICE][MIC][BLOCKED] readable=FAIL"
fi
if test -w "$DEVICE"; then
  echo "[VOICE][MIC] writable=PASS"
else
  echo "[VOICE][MIC][BLOCKED] writable=FAIL"
fi

if command -v fuser >/dev/null 2>&1; then
  echo "[VOICE][MIC] occupancy (fuser):"
  if fuser -v "$DEVICE" 2>&1; then
    echo "[VOICE][MIC][WARN] device appears occupied"
  else
    echo "[VOICE][MIC] device_not_reported_occupied"
  fi
elif command -v lsof >/dev/null 2>&1; then
  echo "[VOICE][MIC] occupancy (lsof):"
  if lsof "$DEVICE" 2>&1; then
    echo "[VOICE][MIC][WARN] device appears occupied"
  else
    echo "[VOICE][MIC] device_not_reported_occupied"
  fi
else
  echo "[VOICE][MIC][SKIPPED] occupancy check requires fuser or lsof"
fi

echo "[VOICE][MIC] ACL changes are intentionally not attempted: /dev ACL support may be unavailable."
echo "[VOICE][MIC] Temporary system workaround: chgrp audio '$DEVICE'; it can be lost after reboot or USB replug."
