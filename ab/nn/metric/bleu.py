from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from ab.nn.metric.caption_text import decoded_batch

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
        hypotheses, targets = decoded_batch(preds, labels)
        for hyp, references in zip(hypotheses, targets):
            if not references:
                continue
            self.scores1.append(sentence_bleu(references, hyp, weights=(1, 0, 0, 0), smoothing_function=self.smooth))
            self.scores2.append(sentence_bleu(references, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smooth))
            self.scores3.append(sentence_bleu(references, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smooth))
            self.scores4.append(sentence_bleu(references, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth))

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
