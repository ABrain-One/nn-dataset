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
            
        self.reset()

    def reset(self):
        self.scores = []
        self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)

    def __call__(self, preds, labels):
        """
        preds: [B, seq_len, vocab_size] (one-hot) or [B, seq_len] (indices)
        labels: [B, seq_len] (indices)
        """
        if self.idx2word is None:
            # Try to fetch again if not available at init
            self.idx2word = GLOBAL_CAPTION_VOCAB.get('idx2word', None)
            if self.idx2word is None:
                print("[MeteorMetric WARN] idx2word not found in GLOBAL_CAPTION_VOCAB. METEOR score will be 0.")
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
            # Decode to words, skipping PAD (0), SOS, EOS
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
