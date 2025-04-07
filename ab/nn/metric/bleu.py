import torch
from ab.nn.metric.base.base import BaseMetric
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEU(BaseMetric):
    """
    Computes BLEU metric scores for image captioning.
    """
    def __init__(self, out_shape=None):
        # out_shape is not used for BLEU, but included for compatibility
        super().__init__()
        self.smooth = SmoothingFunction().method1
        self.reset()
    
    def reset(self):
        """
        Resets the internal evaluation result to its initial state.
        """
        self.scores = []
    
    def update(self, preds, labels):
        """
        Updates the internal evaluation result for a batch.
        :param preds: Model predictions. Expected shape: [batch, seq_len, vocab_size] (logits).
        :param labels: Ground truth labels. Expected shape: [batch, seq_len].
        """
        # Convert logits to predicted token indices
        pred_tokens = torch.argmax(preds, dim=-1)  # shape: [batch, seq_len]
        pred_tokens = pred_tokens.cpu().tolist()
        labels = labels.cpu().tolist()
        
        # For each predicted caption and its reference,
        # compute the BLEU score.
        # (Here we treat each reference caption as a single reference;
        # in practice, you might have multiple reference captions.)
        for pred, ref in zip(pred_tokens, labels):
            score = sentence_bleu([ref], pred, smoothing_function=self.smooth)
            self.scores.append(score)
    
    def __call__(self, outputs, targets):
        """
        Processes a batch and returns dummy values for compatibility with the accumulation pattern.
        """
        self.update(outputs, targets)
        return 1, 1
    
    def result(self):
        """
        Computes and returns the average BLEU score.
        """
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

def create_metric(out_shape=None):
    return BLEU(out_shape)
