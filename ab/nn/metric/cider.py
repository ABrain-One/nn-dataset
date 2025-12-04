import torch

# COCO caption evaluation: CIDEr metric
try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    Cider = None


class CIDErMetric:
    """
    Corpus-level CIDEr metric for image captioning.
    - preds: logits [batch, seq, vocab] OR token ids [batch, seq]
    - labels: token ids [batch, seq] OR [batch, n_refs, seq]

    We treat token ids as "word tokens" by stringifying them
    (same idea as BLEU metric in this codebase).
    """
    def __init__(self, out_shape=None):
        self.cider_scorer = Cider() if Cider is not None else None
        self.reset()

    def reset(self):
        # COCO-style structures:
        # gts: {img_id: [{'caption': str}, ...]}
        # res: {img_id: [{'caption': str}]}
        self._gts = {}
        self._res = {}
        self._img_idx = 0

    def __call__(self, preds, labels):
        # Convert predictions to token ids
        if preds.dim() == 3:
            pred_ids = torch.argmax(preds, -1).cpu().tolist()
        elif preds.dim() == 2:
            pred_ids = preds.cpu().tolist()
        else:
            raise ValueError(f"Preds shape not supported for CIDErMetric: {preds.shape}")

        # References format similar to BLEUMetric
        if labels.dim() == 3:
            targets = labels.cpu().tolist()
        else:
            targets = [[t] for t in labels.cpu().tolist()]

        for p, refs in zip(pred_ids, targets):
            # Remove padding (id == 0)
            hyp = [w for w in p if w != 0]
            filtered_refs = [[w for w in r if w != 0] for r in refs]
            filtered_refs = [ref for ref in filtered_refs if len(ref) > 0]

            if not filtered_refs:
                print("[CIDErMetric WARN] Empty reference for sample; this should not happen often.")
                continue

            if self.cider_scorer is None:
                # If scorer is not available, we silently skip; result() will be 0.0
                continue

            img_id = self._img_idx
            self._img_idx += 1

            # Token ids as string tokens
            hyp_str = " ".join(str(w) for w in hyp)
            ref_strs = [" ".join(str(w) for w in r) for r in filtered_refs]

            self._res[img_id] = [hyp_str]
            self._gts[img_id] = ref_strs

    def result(self):
        """
        Main scalar metric for Optuna/pipelines.
        """
        if self.cider_scorer is None:
            return 0.0
        if not self._res:
            return 0.0
        try:
            score, _ = self.cider_scorer.compute_score(self._gts, self._res)
            return float(score)
        except Exception as e:
            print(f"[CIDErMetric WARN] CIDEr computation failed: {e}")
            return 0.0

    def get_all(self):
        return {
            "CIDEr": self.result()
        }


def create_metric(out_shape=None):
    return CIDErMetric(out_shape)
