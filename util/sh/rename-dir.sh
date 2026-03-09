#!/bin/bash

for dir in *_celeba-gender_acc_rl-init-*; do
    # Construct target name
    target=$(echo "$dir" | sed 's/_celeba-gender_acc_rl-init-/_celeba-gender_acc_alt-/')

    if [ "$dir" != "$target" ]; then
        echo "Processing: $dir → $target"

        if [ -d "$target" ]; then
            echo "⚠️  Target exists: $target — merging contents."
            rsync -a "$dir/" "$target/"
            rm -rf "$dir"
        else
            mv "$dir" "$target"
        fi
    fi
done