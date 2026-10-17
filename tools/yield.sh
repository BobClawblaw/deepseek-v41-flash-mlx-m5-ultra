#!/bin/bash
# Pause this recipe's long jobs (converter, CPU reference) while another session runs a TensorFold engine or a
# benchmark on the Mac; resume them two minutes after it ends.
L=${1:-logs/yield.log}; state=run; quiet=0
jobs() { ps -axo pid=,command= | awk -v me=$$ '$1 != me && /[Pp]ython/ && /(tools\/convert\.py|ref\/study\.py|study\.py (replies|logits))/ {print $1}' | tr '\n' ' '; }
while true; do
  mine=$(jobs)
  [ -z "$mine" ] && { echo "$(date +%T) my jobs finished" >> "$L"; exit 0; }
  others=$(ps -axo pid=,comm= | awk '$2 ~ /(^|\/)(tensorfold|tensorfold-native)$|bench/ {print $1}' | tr '\n' ' ')
  if [ -n "$others" ]; then
    quiet=0
    [ $state = run ] && { kill -STOP $mine; state=paused; echo "$(date +%T) paused $mine (others: $others)" >> "$L"; }
  else
    quiet=$((quiet+1))
    [ $state = paused ] && [ $quiet -ge 4 ] && { kill -CONT $(jobs); state=run; echo "$(date +%T) resumed" >> "$L"; }
  fi
  sleep 30
done
