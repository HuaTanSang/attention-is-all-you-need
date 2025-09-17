from typing import List 
from torchmetrics.functional.text.ter import ter

from torchtext.data.metrics import bleu_score

def BLEU(candidate_corpus: List[str], references_corpus: List[str]): 
    score = bleu_score(candidate_corpus=candidate_corpus, 
                       references_corpus=references_corpus)
    return score


def TER(candidate_corpus: List[str], references_corpus: List[str]): 
    score = ter(candidate_corpus=candidate_corpus, 
                references_corpus=references_corpus)
    return score 