#!/bin/sh
# pe62 scheduler: runs each line of a command file, at most 2 pe62 python jobs at once.
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  while [ $(pgrep -f "^python3 pe62_c" | wc -l) -ge 2 ]; do sleep 10; done
  sh -c "$cmd" &
  sleep 8
done < "$1"
wait
