import os
from metrics.collection.BertScore import BertScore
from xai.Explainer import Explainer
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc
from xai.util.corpus_explainer import explain_corpus
from project_root import ROOT_DIR


class BSExplainer(Explainer):
    '''
    Class to generate reference-free feature importance explanations on a corpus using customized BertScore originally by
    Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. “BERTScore: Evaluating
    Text Generation with BERT”. In: International Conference on Learning Representations. 2020. url:
    https://openreview.net/forum?id=SkeHuCVFDr.
    '''

    # Only works with custom Bertscore enabled in the Bertscore Class! metrics/collection/BertScore.py
    def __init__(self):
        self.BS = None

    def apply_explanation(self, df, metrics=None, recover=False,
                          models=[(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'vicgalle/xlm-roberta-large-xnli-anli'),
                                  (17, 'xlm-roberta-large')]):
        '''
        :param df: The corpus to explain on
        :param metrics: A list of metric names to explain. When none, all metrics known by MetricWrapper will be explained.
        :param recover: Whether to recover previous explanations
        :param models: A list of tuples (layer, model_name) that should be ensembled via summation
        :return: A dictionary with explanation results, that can be interpreted by xai/evaluation/eval4nlp_evaluate.py
        '''
        lp_list = df['LP'].tolist()
        ref_list = df['REF'].tolist()
        hyp_list = df['HYP'].tolist()
        src_list = df['SRC'].tolist()

        # define the BertScore scorer with the first tuple of layer and modelname passed as an input
        self.BS = BertScore(num_layers=models[0][0], model=models[0][1], custom_bert_scorer=True)
        print(self.BS.scorer.hash)

        # get attributions
        scores, hyp_explanations, src_explanations = self.BS(src_list, hyp_list)

        # Get the attributions of further models and sum up the token-level and sentence-level scores
        # Note that this assumes aligned tokenization. For more freedom in the choice of model, it could be done
        # after alignin the words with the original hypothesis
        ensemble_models = models[1:]
        for ensemble in ensemble_models:
            self.BS = BertScore(num_layers=ensemble[0], model=ensemble[1], custom_bert_scorer=True)
            print(self.BS.scorer.hash)
            scores2, hyp_explanations2, src_explanations2 = self.BS(src_list, hyp_list)

            hyp_explanations = [[(e_w[0] + e_w2[0], e_w[1]) for e_w, e_w2 in zip(e, e2)] for e, e2 in
                                zip(hyp_explanations, hyp_explanations2)]
            src_explanations = [[(e_w[0] + e_w2[0], e_w[1]) for e_w, e_w2 in zip(e, e2)] for e, e2 in
                                zip(src_explanations, src_explanations2)]
            scores = [(s + s2) for s, s2 in zip(scores, scores2)]

        def collapse_list(l):
            # Collapses the list of tokens and attributions scores, if subword tokenization denotes spaces with '▁' or 'Ġ'
            # Word-level scores of tokens that belong to the same word are averaged.
            # There could be rare cases where subword tokens contain a space in the middle instead of the beginning of a word.
            # In such a case the word level score might be inferred by the aligning function that is used during evaluation
            # I did not notice such a case in the shared task

            words = [l[0][1][1:]]   # A list of tokens. The first letter of the first token is removed
            scores = [l[0][0]]      # A list of scores per token, initialised with the first one
            inb = 1                 # A counter that indicates over how many elements subwords should be averaged
            for x in range(1, len(l)):
                if l[x][1][0] != '▁' and l[x][1][0] != 'Ġ':  # If words do not start with a spacem I collapse them with the previous
                    words[-1] += l[x][1]  # Appending the last string
                    scores[-1] += l[x][0] # Adding to the last score
                    inb += 1              # Increasing the number of scores that are averaged
                    if len(l) - 1 == x:
                        scores[-1] /= inb # If we have reached the last element, we average
                else:
                    scores[-1] /= inb           # Average with the current inb counter
                    inb = 1                     # Reset the counter
                    words.append(l[x][1][1:])   # Add a new word and remove the trailing space
                    scores.append(l[x][0])      # Add a new score
            return list(zip(words, scores))

        def collapse_list_hash(l):
            # Collapses list of tokens and attribution scores, where subwords begin with ##
            words = [l[0][1]]   # Initialize with first word and score
            scores = [l[0][0]]
            inb = 1
            for x in range(1, len(l)):
                if l[x][1][0:2] == '##':        #If the current word is a subword, it is appended to the previous
                    words[-1] += l[x][1][2:]
                    scores[-1] += l[x][0]       #Their scores are added and the inb counter is increased
                    inb += 1
                    if len(l) - 1 == x:
                        scores[-1] /= inb       # Average at end of sentence
                else:
                    # Remove the underscore
                    scores[-1] /= inb           # Average at next word
                    inb = 1
                    words.append(l[x][1])
                    scores.append(l[x][0])
            return list(zip(words, scores))

        # Remove begin and end tokens and collapse where necessary
        # The if statement contains a hacky way to determine whether ## tokenization was used
        if '##' in ''.join([''.join([e[1] for e in hyp_explanations[x]]) for x in range(len(hyp_explanations))]):
            hyp_explanations = [collapse_list_hash(e[1:-1]) for e in hyp_explanations]
            src_explanations = [collapse_list_hash(e[1:-1]) for e in src_explanations]
        else:
            hyp_explanations = [collapse_list(e[1:-1]) for e in hyp_explanations]
            src_explanations = [collapse_list(e[1:-1]) for e in src_explanations]

        # explanations2 = []
        # for e in hyp_explanations:
        #    m = sum([e_w[1] for e_w in e])/len(e)
        #    explanations2.append([(e_w[0],(e_w[1]-m)) for e_w in e])
        # explanations = explanations2

        attributions = []
        for x in range(len(hyp_explanations)):
            attributions.append({'src': src_list[x], 'ref': ref_list[x], 'hyp': hyp_list[x],
                                 'metrics': {'BERTSCORE': {'attributions': hyp_explanations[x],
                                                           'src_attributions': src_explanations[x], 'score': scores[x]}}
                                 })

        return attributions


if __name__ == '__main__':
    BSE = BSExplainer()
    n = 1000
    et_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_dev_et_en.tsv')
    ro_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_dev_ro_en.tsv')

    explain_corpus(BSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_dev_attributions_bs.json',
                   mlqe_pandas_path=et_en_path,
                   models=[(16, 'joeddav/xlm-roberta-large-xnli'),(17, 'joeddav/xlm-roberta-large-xnli'), (17, 'vicgalle/xlm-roberta-large-xnli-anli'),
                           (21, 'xlm-roberta-large')])


    evaluate_mlqe_auc([
        os.path.join(ROOT_DIR,'xai/output/explanations/0_999_mlqe_et_dev_attributions_bs.json')],
        start=0, end=n, invert=True, f=20, corpus_path=et_en_path)



    '''
    Further configurations previously used
    explain_corpus(BSE,
                   recover=False,
                   from_row=0, to_row=100,
                   outfile='mlqe_et_en_attributions_bs_x.json',
                   models=[(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'joeddav/xlm-roberta-large-xnli'),
                           (17, 'vicgalle/xlm-roberta-large-xnli-anli'),
                           (21, 'xlm-roberta-large')])

    evaluate_mlqe_auc([
        os.path.join(ROOT_DIR,'xai/output/explanations/0_99_mlqe_et_en_attributions_bs_x.json')],
        start=0, end=100, invert=True, f=20)
        
    explain_corpus(BSE,
               recover=False,
               from_row=0, to_row=1000,
               outfile='et_en_test_attributions_bs_xlmr_ensemble',
               models=[(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'joeddav/xlm-roberta-large-xnli'),
                           (17, 'vicgalle/xlm-roberta-large-xnli-anli'),
                           (21, 'xlm-roberta-large')],
               mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_test_et-en.tsv'))

    explain_corpus(BSE,
                   recover=False,
                   from_row=0, to_row=1000,
                   outfile='ro_en_test_attributions_bs_xlmr_ensemble',
                   models=[(16, 'joeddav/xlm-roberta-large-xnli'), (17, 'joeddav/xlm-roberta-large-xnli'),
                           (17, 'vicgalle/xlm-roberta-large-xnli-anli'),
                           (21, 'xlm-roberta-large')],
                   mlqe_pandas_path=os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_test_ro-en.tsv'))

    '''
