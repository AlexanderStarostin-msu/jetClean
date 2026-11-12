#!/bin/bash

echo "Running Allrun in all case folders..."

for d in */ ; do
    if [[ -f "$d/Allrun" ]]; then
        echo ">>> Running $d"
        (cd "$d" && ./Allrun)
    fi
done
