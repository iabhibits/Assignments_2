import numpy as np
import pandas as pd

from nltk.corpus import bleu_corpus()
from evaluate import evaluate_model

def evaluation(model,src,src_test,trg,trg_test,config)
	bleu = evaluate_model(
		    model, src, src_test, trg,
		    trg_test, config, verbose=False,
		    metric='bleu',
	)