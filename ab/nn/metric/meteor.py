import os
import re
from typing import Any

import nltk
import torch
from nltk.translate.meteor_score import meteor_score

from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB


def _resource_exists(*paths):
    for path in paths:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


def _safe_download_nltk() -> None:
    """
    Safely download NLTK resources using a cross-process file lock.
    This ensures that multiple workers/GPUs don't corrupt the download
    by downloading simultaneously, while keeping backward compatibility
    for legacy scripts that depend on automatic downloads.
    """
    from filelock import FileLock, Timeout
    from pathlib import Path

    # Fast path: check if resources already exist
    if _resource_exists("corpora/wordnet", "corpora/wordnet.zip") and \
       _resource_exists("corpora/omw-1.4", "corpora/omw-1.4.zip"):
        return

    # Slow path: use a lock to download safely
    import tempfile
    lock_dir = Path(tempfile.gettempdir()) / "nltk-download-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "meteor_nltk.lock"

    try:
        with FileLock(str(lock_path), timeout=600):
            # Double check inside the lock
            if not _resource_exists("corpora/wordnet", "corpora/wordnet.zip"):
                success = nltk.download('wordnet', quiet=True)
                if not success:
                    raise RuntimeError("Failed to download NLTK resource: wordnet")

            if not _resource_exists("corpora/omw-1.4", "corpora/omw-1.4.zip"):
                success = nltk.download('omw-1.4', quiet=True)
                if not success:
                    raise RuntimeError("Failed to download NLTK resource: omw-1.4")
    except Timeout:
        if not (
            _resource_exists("corpora/wordnet", "corpora/wordnet.zip")
            and _resource_exists("corpora/omw-1.4", "corpora/omw-1.4.zip")
        ):
            print(
                "[MeteorMetric WARN] Timed out waiting for NLTK download lock "
                "and required resources are still unavailable. "
                "METEOR scores may be skipped or reported as 0."
            )


def _tokenize_text(text: str) -> list[str]:
    """
    Tokenize and normalize English caption text.

    Falls back to whitespace tokenization if NLTK punkt resources are missing.
    """
    text = str(text).lower().strip()

    try:
        from nltk.tokenize import word_tokenize
        words = word_tokenize(text)
    except LookupError:
        words = text.split()

    cleaned_words = [
        re.sub(r"[^a-z0-9]", "", word)
        for word in words
    ]

    return [word for word in cleaned_words if word]


class MeteorMetric:
    """
    METEOR metric for image captioning.

    Supported prediction formats:
    - list[str]
    - token IDs with shape [batch, sequence]
    - logits with shape [batch, sequence, vocabulary]

    Supported label formats:
    - [batch, sequence]
    - [batch, references, sequence]
    """

    SPECIAL_TOKENS = {
        "<SOS>",
        "<EOS>",
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<|endoftext|>",
    }

    def __init__(self, out_shape=None):
        _safe_download_nltk()

        self.vocab_size = int(out_shape[0]) if out_shape else 0
        self.gpt2_tokenizer = None
        self.warned_missing_vocab = False

        if self.vocab_size == 50257:
            try:
                from transformers import GPT2TokenizerFast

                tokenizer_dir = os.path.join(
                    os.path.dirname(__file__),
                    "../transform/gpt2_tokenizer",
                )

                self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(
                    tokenizer_dir,
                    local_files_only=True,
                )

            except (ImportError, OSError) as error:
                print(
                    "[MeteorMetric WARN] Local GPT-2 tokenizer could not be "
                    f"loaded. Falling back to GLOBAL_CAPTION_VOCAB. Error: {error}"
                )
                self.gpt2_tokenizer = None

        self.reset()

    def reset(self):
        self.scores = []
        self.idx2word = GLOBAL_CAPTION_VOCAB.get("idx2word")

    def _get_idx2word(self):
        if self.idx2word is None:
            self.idx2word = GLOBAL_CAPTION_VOCAB.get("idx2word")

        if self.idx2word is None:
            if not self.warned_missing_vocab:
                print(
                    "[MeteorMetric WARN] idx2word was not found in "
                    "GLOBAL_CAPTION_VOCAB. Samples requiring COCO-vocabulary "
                    "decoding will be skipped."
                )
                self.warned_missing_vocab = True

            return None

        return self.idx2word

    def _decode_legacy_ids(self, token_ids: list[int]) -> list[str]:
        """
        Decode IDs using the repository's COCO/GPT-2-compatible idx2word object.

        Uses .get() intentionally so dictionary-like wrapper objects remain
        compatible.
        """
        idx2word = self._get_idx2word()

        if idx2word is None:
            return []

        words = []

        for token_id in token_ids:
            try:
                token_id = int(token_id)
            except (TypeError, ValueError):
                continue

            if token_id in {0, -100}:
                continue

            word = idx2word.get(token_id, "")

            if word is None:
                continue

            word = str(word).strip()

            if not word or word in self.SPECIAL_TOKENS:
                continue

            words.append(word.lower())

        return words

    def _decode_gpt2_ids(self, token_ids: list[int]) -> list[str]:
        if self.gpt2_tokenizer is None:
            return self._decode_legacy_ids(token_ids)

        clean_ids = []

        for token_id in token_ids:
            try:
                token_id = int(token_id)
            except (TypeError, ValueError):
                continue

            if token_id >= 0 and token_id != -100:
                clean_ids.append(token_id)

        if not clean_ids:
            return []

        text = self.gpt2_tokenizer.decode(
            clean_ids,
            skip_special_tokens=True,
        )

        return _tokenize_text(text)

    def _decode_ids(self, token_ids: list[int]) -> list[str]:
        if self.vocab_size == 50257 and self.gpt2_tokenizer is not None:
            return self._decode_gpt2_ids(token_ids)

        return self._decode_legacy_ids(token_ids)

    @staticmethod
    def _prepare_targets(labels: torch.Tensor) -> list[list[list[int]]]:
        """
        Convert labels into:

            batch -> references -> token IDs
        """
        if labels.dim() == 2:
            return [
                [reference]
                for reference in labels.detach().cpu().tolist()
            ]

        if labels.dim() == 3:
            return labels.detach().cpu().tolist()

        raise ValueError(
            "Labels shape not supported by MeteorMetric: "
            f"{tuple(labels.shape)}"
        )

    @staticmethod
    def _prepare_predictions(preds: torch.Tensor) -> list[list[int]]:
        if preds.dim() == 3:
            return torch.argmax(
                preds,
                dim=-1,
            ).detach().cpu().tolist()

        if preds.dim() == 2:
            return preds.detach().cpu().tolist()

        raise ValueError(
            "Predictions shape not supported by MeteorMetric: "
            f"{tuple(preds.shape)}"
        )

    def _append_score(
        self,
        hypothesis_words: list[str],
        reference_words: list[list[str]],
    ) -> None:
        reference_words = [
            reference
            for reference in reference_words
            if reference
        ]

        if not hypothesis_words or not reference_words:
            return

        try:
            score = meteor_score(
                reference_words,
                hypothesis_words,
            )
        except Exception as error:
            print(
                "[MeteorMetric WARN] METEOR calculation failed for one "
                f"sample and was skipped. Error: {error}"
            )
            return

        self.scores.append(float(score))

    def __call__(self, preds: Any, labels: torch.Tensor):
        target_ids = self._prepare_targets(labels)

        # ---------------------------------------------------------------------
        # Text predictions, used by models such as GIT
        # ---------------------------------------------------------------------
        if isinstance(preds, list):
            if not preds:
                return
            if not all(isinstance(item, str) for item in preds):
                raise TypeError("MeteorMetric expected list[str] predictions.")
            for hypothesis_text, sample_references in zip(preds, target_ids):
                hypothesis_words = _tokenize_text(hypothesis_text)

                reference_words = [
                    self._decode_ids(reference_ids)
                    for reference_ids in sample_references
                ]

                self._append_score(
                    hypothesis_words,
                    reference_words,
                )

            return

        # ---------------------------------------------------------------------
        # Tensor predictions
        # ---------------------------------------------------------------------
        if not torch.is_tensor(preds):
            raise TypeError(
                "MeteorMetric predictions must be a tensor or list[str], "
                f"received: {type(preds).__name__}"
            )

        prediction_ids = self._prepare_predictions(preds)

        for predicted_ids, sample_references in zip(
            prediction_ids,
            target_ids,
        ):
            hypothesis_words = self._decode_ids(predicted_ids)

            reference_words = [
                self._decode_ids(reference_ids)
                for reference_ids in sample_references
            ]

            self._append_score(
                hypothesis_words,
                reference_words,
            )

    def result(self) -> float:
        if not self.scores:
            return 0.0

        return float(sum(self.scores) / len(self.scores))

    def __str__(self) -> str:
        return "METEOR"


def create_metric(out_shape=None):
    return MeteorMetric(out_shape)
