"""Decoded-word METEOR metric for single- and multi-reference captions."""

import nltk
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score

from ab.nn.metric.caption_text import decoded_batch


class MeteorMetric:
    def __init__(self, out_shape=None):
        del out_shape
        try:
            wordnet.ensure_loaded()
            self.wordnet = wordnet
        except LookupError:
            # Use the same metric definition on a fresh installation.
            nltk.download("wordnet", quiet=True, raise_on_error=True)
            wordnet.ensure_loaded()
            self.wordnet = wordnet
        self.reset()

    def reset(self):
        self.scores = []

    def __call__(self, predictions, labels):
        hypotheses, references = decoded_batch(predictions, labels)
        for hypothesis, sample_references in zip(hypotheses, references):
            if not hypothesis:
                self.scores.append(0.0)
                continue
            self.scores.append(
                meteor_score(
                    sample_references,
                    hypothesis,
                    wordnet=self.wordnet,
                )
            )

    def result(self):
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def __str__(self):
        return "METEOR"


def create_metric(out_shape=None):
    return MeteorMetric(out_shape)
