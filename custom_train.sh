#!/bin/bash

# This script runs the training process for a specified configuration.
# Example usage: ./custom_train.sh

# Activate virtual environment if it exists
if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Run the training for a sample configuration
python -m ab.nn.train -c img-classification_cifar-10_acc_AlexNet -e 1 -t 1
