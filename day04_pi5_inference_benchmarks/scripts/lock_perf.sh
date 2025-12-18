#!/usr/bin/env bash
set -euo pipefail

echo "[lock_perf] enabling performance mode..."

# Require sudo
if [[ $EUID -ne 0 ]]; then
  echo "Run as: sudo $0"
  exit 1
fi

# 1) CPU governor -> performance
for cpu_gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -f "$cpu_gov" ]] && echo performance > "$cpu_gov"
done

# 2) Disable CPU idle deep states (reduces latency jitter)
# (If file exists; some kernels differ)
for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
  [[ -f "$state" ]] && echo 1 > "$state"
done

# 3) Reduce kernel printk noise
sysctl -w kernel.printk="3 3 3 3" >/dev/null

# 4) Make sure we aren’t thermally throttling immediately
vcgencmd measure_temp || true
vcgencmd get_throttled || true

echo "[lock_perf] done."
