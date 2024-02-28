#!/usr/bin/env bash
echo 'kill existed process'
bash kill.sh

echo 'start a new process'

hostname='unknown'
if [ -f /root/.ssh/hostname ]; then
    hostname=$(cat /root/.ssh/hostname)
fi

log="$hostname.log"
echo $log
rm $log

nohup python -u main_g.py >$log 2>&1 &

tail -f $log -n 9999
