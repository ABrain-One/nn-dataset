import re
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BLEUMetric:
    def __init__(self, out_shape=None):
        self.smooth = SmoothingFunction().method1
        self.vocab_size = out_shape[0] if out_shape else 0
        if self.vocab_size == 50257:
            try:
                from transformers import GPT2TokenizerFast
                self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            except ImportError:
                self.gpt2_tokenizer = None
        else:
            self.gpt2_tokenizer = None
        self.reset()



    def reset(self):
        self.scores1 = []  # BLEU-1
        self.scores2 = []  # BLEU-2
        self.scores3 = []  # BLEU-3
        self.scores4 = []  # BLEU-4

    def __call__(self, preds, labels):
        if isinstance(preds, list) and isinstance(preds[0], str):
            if labels.dim() == 3:
                targets = labels.cpu().tolist()
            else:
                targets = [[t] for t in labels.cpu().tolist()]
            
            from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB
            idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
            
            for hyp_text, refs in zip(preds, targets):
                try:
                    from nltk.tokenize import word_tokenize
                    hyp = [re.sub(r'[^a-z0-9]', '', w) for w in word_tokenize(hyp_text.lower())]
                except Exception:
                    import re
                    hyp = [re.sub(r'[^a-z0-9]', '', w) for w in hyp_text.lower().split()]
                hyp = [w for w in hyp if w]
                
                filtered_refs = []
                for r in refs:
                    r_clean = [idx2word.get(x, "") for x in r if x != 0]
                    r_clean = [w.lower() for w in r_clean if w and w not in ('<EOS>', '<SOS>', '<PAD>', '<UNK>', '<|endoftext|>')]
                    if len(r_clean) > 0:
                        filtered_refs.append(r_clean)
                if not filtered_refs:
                    continue
                self.scores1.append(sentence_bleu(filtered_refs, hyp, weights=(1, 0, 0, 0), smoothing_function=self.smooth))
                self.scores2.append(sentence_bleu(filtered_refs, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smooth))
                self.scores3.append(sentence_bleu(filtered_refs, hyp, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smooth))
                self.scores4.append(sentence_bleu(filtered_refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth))
            return

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
            if self.vocab_size == 50257 and self.gpt2_tokenizer:
                # NEW LOGIC: Text-based Decoding (GPT-2/OPT)
                p_clean = [x for x in p if x != -100 and x >= 0]
                hyp_text = self.gpt2_tokenizer.decode(p_clean, skip_special_tokens=True)
                hyp = hyp_text.lower().split()
                filtered_refs = []
                for r in refs:
                    r_clean = [x for x in r if x != -100 and x >= 0]
                    ref_text = self.gpt2_tokenizer.decode(r_clean, skip_special_tokens=True)
                    if ref_text.strip():
                        filtered_refs.append(ref_text.lower().split())
            else:
                # LEGACY LOGIC: Integer ID-based Lookup
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
