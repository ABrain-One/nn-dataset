#!/bin/bash

if [ -d .venv ]; then
    source .venv/bin/activate
fi

python -m ab.nn.train "$@"