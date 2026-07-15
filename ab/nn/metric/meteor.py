import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from ab.nn.loader.coco_.Caption import GLOBAL_CAPTION_VOCAB

class MeteorMetric:
    """
    METEOR Metric (Metric for Evaluation of Translation with Explicit ORdering)
    """
    def __init__(self, out_shape=None):
        # Ensure NLTK data is downloaded
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('corpora/omw-1.4.zip')
        except LookupError:
            nltk.download('omw-1.4')
            
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
        self.scores = []
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)

    def __call__(self, preds, labels):
        """
        preds: [B, seq_len, vocab_size] (one-hot) or [B, seq_len] (indices) or list of strings
        labels: [B, seq_len] (indices)
        """
        if isinstance(preds, list) and isinstance(preds[0], str):
            if labels.dim() == 2:
                target_ids = [[t] for t in labels.cpu().tolist()]
            elif labels.dim() == 3:
                target_ids = labels.cpu().tolist()
            
            self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', {})
            for hyp_text, t_list in zip(preds, target_ids):
                try:
                    from nltk.tokenize import word_tokenize
                    import re
                    hyp_words = [re.sub(r'[^a-z0-9]', '', w) for w in word_tokenize(hyp_text.lower())]
                except Exception:
                    import re
                    hyp_words = [re.sub(r'[^a-z0-9]', '', w) for w in hyp_text.lower().split()]
                hyp_words = [w for w in hyp_words if w]
                
                ref_list_words = []
                for t in t_list:
                    ref_words = [self.idx2word[i] for i in t if i in self.idx2word and i != 0 and self.idx2word.get(i, '') not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
                    if ref_words:
                        ref_list_words.append(ref_words)
                if not ref_list_words:
                    continue
                score = meteor_score(ref_list_words, hyp_words)
                self.scores.append(score)
            return

        if self.vocab_size != 50257:
            if self.idx2word is None:
                # Try to fetch again if not available at init
                self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)
                if self.idx2word is None:
                    if not getattr(self, 'warned_missing_vocab', False):
                        print("[MeteorMetric WARN] idx2word not found in GLOBAL_CAPTION_VOCAB. METEOR score will be 0.")
                        self.warned_missing_vocab = True
                    return

        # Convert preds to indices if needed
        if preds.dim() == 3:
            pred_ids = torch.argmax(preds, -1).cpu().tolist()
        elif preds.dim() == 2:
            pred_ids = preds.cpu().tolist()
        else:
            raise ValueError(f"Preds shape not supported: {preds.shape}")

        # Convert labels to list
        # Convert labels to list
        if labels.dim() == 2:
            target_ids = [[t] for t in labels.cpu().tolist()]
        elif labels.dim() == 3:
            target_ids = labels.cpu().tolist()
        else:
            raise ValueError(f"Labels shape not supported: {labels.shape}")

        for p, t_list in zip(pred_ids, target_ids):
            if self.vocab_size == 50257 and self.gpt2_tokenizer:
                # NEW LOGIC: Text-based Decoding (GPT-2/OPT)
                p_clean = [x for x in p if x != -100 and x >= 0]
                hyp_text = self.gpt2_tokenizer.decode(p_clean, skip_special_tokens=True)
                hyp_words = hyp_text.lower().split()
                
                ref_list_words = []
                for t in t_list:
                    t_clean = [x for x in t if x != -100 and x >= 0]
                    ref_text = self.gpt2_tokenizer.decode(t_clean, skip_special_tokens=True)
                    if ref_text.strip():
                        ref_list_words.append(ref_text.lower().split())
            else:
                # LEGACY LOGIC: Integer ID-based Lookup
                hyp_words = [self.idx2word[i] for i in p if i in self.idx2word and i != 0 and self.idx2word[i] not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
                
                ref_list_words = []
                for t in t_list:
                    ref_words = [self.idx2word[i] for i in t if i in self.idx2word and i != 0 and self.idx2word[i] not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
                    if ref_words:
                        ref_list_words.append(ref_words)
            
            if not ref_list_words:
                continue
                
            score = meteor_score(ref_list_words, hyp_words)
            self.scores.append(score)

    def result(self):
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __str__(self):
        return "METEOR"

def create_metric(out_shape=None):
    return MeteorMetric(out_shape)
