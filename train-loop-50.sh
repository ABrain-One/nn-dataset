#!/bin/bash

# An endless training loop, restarting the program after critical errors.
# All image classification tasks are randomly shuffled and trained over 50 epochs.

bash train-loop.sh -c img-classification -e 50 -r 1