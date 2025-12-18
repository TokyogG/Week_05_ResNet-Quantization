#!/usr/bin/env bash
set -euo pipefail

echo "---- ENV SNAPSHOT ----"
date
uname -a
python3 --version
echo "[cpu]"
lscpu | egrep 'Model name|CPU\(s\)|Thread|MHz|BogoMIPS' || true

echo "[governors]"
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -f "$f" ]] && echo "$f: $(cat "$f")"
done

echo "[freq]"
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
  [[ -f "$f" ]] && echo "$f: $(cat "$f")"
done

echo "[temp/throttle]"
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true
echo "----------------------"