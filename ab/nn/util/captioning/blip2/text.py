"""Caption targets with the same tokenized prefix used during generation."""

import torch


def caption_training_batch(tokenizer, prompt, captions, max_length, device, *, add_special_tokens):
    prefix = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    budget = max_length - len(prefix) - 1  # Keep an explicit end-of-caption target.
    if budget < 1:
        raise ValueError("max_text_length must leave room for the prompt, caption and EOS.")
    rows = []
    for caption in captions:
        tokens = tokenizer(caption.strip(), add_special_tokens=False)["input_ids"][:budget]
        if not tokens:
            raise ValueError("A training caption must contain at least one token.")
        rows.append(prefix + tokens + [tokenizer.eos_token_id])
    if not rows:
        raise ValueError("Cannot train on an empty caption batch.")
    ids = torch.full((len(rows), max(map(len, rows))), tokenizer.pad_token_id,
                     dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    targets = torch.full_like(ids, -100)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[index, :len(row)] = 1
        targets[index, len(prefix):len(row)] = ids[index, len(prefix):len(row)]
    return ids, mask, targets
