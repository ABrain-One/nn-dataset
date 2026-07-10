import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEUMetric:
    def __init__(self, out_shape=None):
        self.smooth = SmoothingFunction().method1
        self.reset()

    def reset(self):
        self.scores1 = []  # BLEU-1
        self.scores2 = []  # BLEU-2
        self.scores3 = []  # BLEU-3
        self.scores4 = []  # BLEU-4

    def __call__(self, preds, labels):
        # Accepts logits [batch, seq, vocab] or token ids [batch, seq]
        if preds.dim() == 3:
            pred_ids = torch.argmax(preds, -1).cpu().tolist()
        elif preds.dim() == 2:
            pred_ids = preds.cpu().tolist()
        else:
            raise ValueError(f"Preds shape not supported for BLEUMetric: {preds.shape}")
        # All references for each sample
        if labels.dim() == 3:
            targets = labels.cpu().tolist()
        else:
            targets = [[t] for t in labels.cpu().tolist()]
        for p, refs in zip(pred_ids, targets):
            hyp = [w for w in p if w != 0]
            filtered_refs = [[w for w in r if w != 0] for r in refs]
            filtered_refs = [ref for ref in filtered_refs if len(ref) > 0]
            if not filtered_refs:
                print("[BLEUMetric WARN] Empty reference for sample; this should not happen often.")
                continue
            self.scores1.append(sentence_bleu(filtered_refs, hyp, weights=(1, 0, 0, 0), smoothing_function=self.smooth))
            self.scores2.append(sentence_bleu(filtered_refs, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smooth))
            self.scores3.append(sentence_bleu(filtered_refs, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smooth))
            self.scores4.append(sentence_bleu(filtered_refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth))

    def result(self):
        # Return BLEU-4 for Optuna/pipeline
        return float(sum(self.scores4)) / max(len(self.scores4), 1)

    def get_all(self):
        return {
            'BLEU-1': float(sum(self.scores1)) / max(len(self.scores1), 1),
            'BLEU-2': float(sum(self.scores2)) / max(len(self.scores2), 1),
            'BLEU-3': float(sum(self.scores3)) / max(len(self.scores3), 1),
            'BLEU-4': float(sum(self.scores4)) / max(len(self.scores4), 1)
        }

def create_metric(out_shape=None):
    return BLEUMetric(out_shape)
