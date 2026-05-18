import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class BLEUMetric:
    def __init__(self, out_shape=None):
        self.smooth = SmoothingFunction().method1
        self.out_shape = out_shape
        self.tokenizer = None
        self.reset()

    def reset(self):
        self.scores1 = []
        self.scores2 = []
        self.scores3 = []
        self.scores4 = []

    def _maybe_get_gpt2_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            self.tokenizer = False
        return self.tokenizer if self.tokenizer is not False else None

    def _clean_ids(self, ids):
        # Works for old COCO vocab and GPT2 cached captions.
        bad = {0, 1, 2, -100, 50256}
        return [int(x) for x in ids if int(x) not in bad]

    def _decode_or_ids(self, ids):
        ids = self._clean_ids(ids)

        # GPT2 vocab mode: out_shape is usually (50257,) or ids contain GPT2 range.
        use_gpt2 = False
        if self.out_shape and len(self.out_shape) > 0:
            try:
                use_gpt2 = int(self.out_shape[0]) == 50257 
            except Exception:
                use_gpt2 = False
        if ids and max(ids) > 30000:
            use_gpt2 = True

        if use_gpt2:
            tok = self._maybe_get_gpt2_tokenizer()
            if tok is not None:
                text = tok.decode(ids, skip_special_tokens=True)
                # Strip punctuation for fair evaluation against COCO ground truth
                import string
                text = text.translate(str.maketrans('', '', string.punctuation))
                words = text.lower().strip().split()
                if len(self.scores4) % 100 == 0: # Print every 100th sample
                    print(f"DEBUG BLEU: IDs={ids[:5]}... -> Words={words}")
                return words if words else [str(i) for i in ids]

        # Backward-compatible old behavior: token ids as symbols.
        return [str(i) for i in ids]

    def __call__(self, preds, labels):
        if preds.dim() == 3:
            pred_ids = torch.argmax(preds, -1).detach().cpu().tolist()
        elif preds.dim() == 2:
            pred_ids = preds.detach().cpu().tolist()
        else:
            raise ValueError(f"Preds shape not supported for BLEUMetric: {preds.shape}")

        if labels.dim() == 3:
            targets = labels.detach().cpu().tolist()
        else:
            targets = [[t] for t in labels.detach().cpu().tolist()]

        for p, refs in zip(pred_ids, targets):
            hyp = self._decode_or_ids(p)
            filtered_refs = [self._decode_or_ids(r) for r in refs]
            filtered_refs = [ref for ref in filtered_refs if len(ref) > 0]

            if not filtered_refs or not hyp:
                continue

            self.scores1.append(sentence_bleu(
                filtered_refs, hyp,
                weights=(1, 0, 0, 0),
                smoothing_function=self.smooth
            ))
            self.scores2.append(sentence_bleu(
                filtered_refs, hyp,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=self.smooth
            ))
            self.scores3.append(sentence_bleu(
                filtered_refs, hyp,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=self.smooth
            ))
            self.scores4.append(sentence_bleu(
                filtered_refs, hyp,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smooth
            ))

    def result(self):
        return float(sum(self.scores4)) / max(len(self.scores4), 1)

    def get_all(self):
        return {
            "BLEU-1": float(sum(self.scores1)) / max(len(self.scores1), 1),
            "BLEU-2": float(sum(self.scores2)) / max(len(self.scores2), 1),
            "BLEU-3": float(sum(self.scores3)) / max(len(self.scores3), 1),
            "BLEU-4": float(sum(self.scores4)) / max(len(self.scores4), 1),
        }


def create_metric(out_shape=None):
    return BLEUMetric(out_shape)
