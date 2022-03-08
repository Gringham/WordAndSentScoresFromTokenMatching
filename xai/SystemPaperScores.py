import os

from project_root import ROOT_DIR
from xai.BSExplainer import BSExplainer
from xai.XMSExplainer import XMSExplainer
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc
from xai.util.corpus_explainer import explain_corpus

BSE = BSExplainer()
n = 1000
et_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_dev_et_en.tsv')
ro_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4NLP/eval4nlp_dev_ro_en.tsv')

# This script contains code to reproduce the scores of the Eval4NLP system paper submission.
# Though the script currently has no sleep or save command included, i.e. the results should be copied fast ;)
# If necessary, this should be relatively easy to integrate


# XLMR
print('XLMR')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(17,'xlm-roberta-large')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)

# NLI1
print('NLI1')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(16, 'joeddav/xlm-roberta-large-xnli')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)

# NLI2
print('NLI2')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(17, 'vicgalle/xlm-roberta-large-xnli-anli')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)

# Ensemble configuration

print('Ensemble')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'vicgalle/xlm-roberta-large-xnli-anli'), (17,'xlm-roberta-large')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)


# MBert configuration
print('MBERT')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(9, 'bert-base-multilingual-cased')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)

# MBart configuration
print('MBART')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=et_en_path,
               models = [(12, 'facebook/mbart-large-50-many-to-many-mmt')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=et_en_path)

# XMS mBert
XMSE = XMSExplainer()
explain_corpus(XMSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_attributions_xms_ens.json',
                   mlqe_pandas_path=et_en_path,
                   models=[(12,'','.map')],
                   xlm = False, drop_punctuation=True,
                   embed ='CLP_1', cat='SUM', k='2')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=et_en_path)

# XMS mBert - keep
XMSE = XMSExplainer()
explain_corpus(XMSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_et_attributions_xms_ens.json',
               mlqe_pandas_path=et_en_path,
               models=[(12, '', '.map')],
               xlm=False, drop_punctuation=False,
               embed='CLP_1', cat='SUM', k='2')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=et_en_path)

# XMS ensemble
XMSE = XMSExplainer()
explain_corpus(XMSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_attributions_xms_ens.json',
                   mlqe_pandas_path=et_en_path,
                   models = [(16, 'joeddav/xlm-roberta-large-xnli', '.map_nli1'), (17, 'vicgalle/xlm-roberta-large-xnli-anli', '.map_nli2'), (17,'xlm-roberta-large', '.map_base')],
                   xlm = True, drop_punctuation=True,
                   embed ='UNIGRAM', cat='SUM', k='30')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=et_en_path)


####################################################################
# XLMR
print('XLMR')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(17,'xlm-roberta-large')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)

# NLI1
print('NLI1')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(16, 'joeddav/xlm-roberta-large-xnli')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)

# NLI2
print('NLI2')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(17, 'vicgalle/xlm-roberta-large-xnli-anli')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)

# Ensemble configuration
print('Ensemble')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'vicgalle/xlm-roberta-large-xnli-anli'), (17,'xlm-roberta-large')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)


# MBert configuration
print('MBERT')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(9, 'bert-base-multilingual-cased')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)

# MBart configuration
print('MBART')
explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_de_dev_attributions_bs.json',
               mlqe_pandas_path=ro_en_path,
               models = [(12, 'facebook/mbart-large-50-many-to-many-mmt')])

evaluate_mlqe_auc([
    os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_de_dev_attributions_bs.json')],
    start=0, end=n, invert=True, f=20,corpus_path=ro_en_path)

# XMS mBert
XMSE = XMSExplainer()
explain_corpus(XMSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_attributions_xms_ens.json',
                   mlqe_pandas_path=ro_en_path,
                   models=[(12,'','.map')],
                   xlm = False, drop_punctuation=True,
                   embed ='CLP_1', cat='SUM', k='2')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=ro_en_path)

# XMS mBert - keep
XMSE = XMSExplainer()
explain_corpus(XMSE,
               recover=False,
               from_row=0, to_row=n,
               outfile='mlqe_et_attributions_xms_ens.json',
               mlqe_pandas_path=ro_en_path,
               models=[(12, '', '.map')],
               xlm=False, drop_punctuation=False,
               embed='CLP_1', cat='SUM', k='2')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=ro_en_path)

# XMS ensemble
XMSE = XMSExplainer()
explain_corpus(XMSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_attributions_xms_ens.json',
                   mlqe_pandas_path=ro_en_path,
                   models = [(16, 'joeddav/xlm-roberta-large-xnli', '.map_nli1'), (17, 'vicgalle/xlm-roberta-large-xnli-anli', '.map_nli2'), (17,'xlm-roberta-large', '.map_base')],
                   xlm = True, drop_punctuation=True,
                   embed ='UNIGRAM', cat='SUM', k='30')

evaluate_mlqe_auc([os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_attributions_xms_ens.json')],
                   start=0, end=n, invert=True, f=100,
                   corpus_path=ro_en_path)