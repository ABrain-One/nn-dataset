#!/bin/bash
cd ~/thesis/nn-dataset

if [ -d .venv ]; then
    source .venv/bin/activate
fi

# Usage: ./run_layer_analysis.sh img-classification_cifar-10_acc
# To save logs: ./run_layer_analysis.sh img-classification_cifar-10_acc 2>&1 | tee out/training_log.txt

python -m ab.nn.train -c "$1" -e 50 --layer_analysis
