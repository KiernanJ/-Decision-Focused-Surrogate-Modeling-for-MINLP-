#!/bin/zsh
# Catches the failure mode that cost 1h51m: a forked Pool whose workers all died
# on an AttributeError while the parent blocked forever on results. Processes
# existed the whole time, so "is it running?" answered yes. Total CPU is what
# actually distinguishes working from hung.
cd "$(dirname "$0")"
low=0
while true; do
  alive=$(pgrep -f "11_vcard.py|00_refs.py|15_proxy.py|16_proxy_self.py|17_cut_ablation.py" | wc -l | tr -d ' ')
  cpu=$(ps -A -o %cpu | awk '{s+=$1} END {printf "%.0f", s}')
  if [[ $alive -gt 0 && $cpu -lt 250 ]]; then
    low=$((low+1))
    if [[ $low -ge 3 ]]; then
      print -r -- "[$(date '+%H:%M:%S')] *** STALL: $alive procs alive but CPU ${cpu}% for 3 samples ***" \
        | tee -a logs/watchdog.log
      low=0
    fi
  else
    low=0
  fi
  [[ $alive -eq 0 ]] && print -r -- "[$(date '+%H:%M:%S')] idle (cpu ${cpu}%)" >> logs/watchdog.log
  sleep 180
done
