#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run as: sudo $0"
  exit 1
fi

echo "[unlock_perf] restoring defaults..."

# governor -> ondemand (or schedutil depending on distro)
for cpu_gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -f "$cpu_gov" ]] && echo ondemand > "$cpu_gov" || true
done

# re-enable idle states
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
  [[ -f "$state" ]] && echo 0 > "$state"
done

echo "[unlock_perf] done."
