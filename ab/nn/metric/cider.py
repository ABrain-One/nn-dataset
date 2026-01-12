
import numpy as np
import torch
from collections import defaultdict
import copy

# Embed the Scorer class directly to ensure self-contained file
class CiderScorer(object):
    """
    CIDEr scorer.
    """
    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(self.cook_refs(refs))
            if test is not None:
                self.ctest.append(self.cook_test(test))
            else:
                # mismatch length
                self.ctest.append(None)

    def size(self):
        return len(self.crefs)

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram)-1
                vec[n][ngram] = float(term_freq)*(self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram,count) in vec_hyp[n].items():
                    if ngram in vec_ref[n]:
                        val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n]*norm_ref[n])
                val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
            return val

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def cook_refs(self, refs, n=4):
        return [self.precook(ref, n) for ref in refs]

    def cook_test(self, test, n=4):
        return self.precook(test, n)

    def precook(self, s, n=4):
        words = s.split()
        counts = defaultdict(int)
        for k in range(1,n+1):
            for i in range(len(words)-k+1):
                ngram = tuple(words[i:i+k])
                counts[ngram] += 1
        return counts

# Wrapper for ab.nn.util module loader
class CiderMetric:
    def __init__(self, out_shape=None):
        self.scorer = CiderScorer(n=4, sigma=6.0)
        self.predictions = []
        self.references = []
        self.idx2word = None

    def reset(self):
        self.predictions = []
        self.references = []
        # Create fresh scorer to reset doc freqs etc if needed
        self.scorer = CiderScorer(n=4, sigma=6.0)

    def set_vocab(self, idx2word):
        self.idx2word = idx2word

    def decode(self, ids):
        if self.idx2word is None:
            # Fallback for when idx2word is not set
            return " ".join([str(i) for i in ids])
        
        words = []
        for idx in ids:
            w = self.idx2word.get(idx, '')
            if w in ['<EOS>', '<PAD>', '<BOS>', '<UNK>']:
                 continue
            words.append(w)
        return " ".join(words)

    def __call__(self, preds, labels):
        # preds: [B, T, V] or [B, T]
        if preds.dim() == 3:
            pred_ids = torch.argmax(preds, -1).cpu().tolist()
        else:
            pred_ids = preds.cpu().tolist()
            
        if labels.dim() == 3:
            # Multi-ref or just reshaped? Usually [B, NumRef, T]
            # But standard loader gives [B, T] for single caption training
            # We will handle [B, T]
            label_ids = labels[:, 0, :].cpu().tolist()
        else:
            label_ids = labels.cpu().tolist()

        # We assume idx2word is available via dependency injection or we need to pass it
        # For now, we will store raw IDs if we can't decode, but Cider needs strings.
        # This assumes the caller will set vocab or we use simple space-joined IDs as tokens (works for stats)
        
        for p, t in zip(pred_ids, label_ids):
            # If idx2word is not set, we can't really compute valid CIDEr unless we treat IDs as words
            # Treat IDs as words is valid for n-gram overlap
            
            hyp_str = " ".join(str(x) for x in p if x != 0) # 0 is usually PAD
            ref_str = " ".join(str(x) for x in t if x != 0)
            
            self.predictions.append(hyp_str)
            self.references.append([ref_str]) # List of refs

    def result(self):
        if not self.predictions:
             return 0.0
        
        # Add all to scorer
        # self.scorer.compute_doc_freq() is usually done on Train set refs for CIDEr-D
        # Here we do it batch-wise or epoch-wise? Standard COCO eval usually uses full train set DF.
        # For simple validation metric, we compute DF on current references. 
        
        self.scorer = CiderScorer(n=4, sigma=6.0)
        for hyp, refs in zip(self.predictions, self.references):
            self.scorer.cook_append(hyp, refs)
            
        self.scorer.compute_doc_freq() 
        scores = self.scorer.compute_cider()
        return np.mean(scores)

def create_metric(out_shape=None):
    return CiderMetric(out_shape)

