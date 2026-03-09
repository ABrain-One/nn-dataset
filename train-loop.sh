#!/bin/bash

# An endless training loop, restarting the program after critical errors

if [ -d .venv ]; then
    source .venv/bin/activate
fi

while true ; do python -m ab.nn.train "$@"; done