import os
import dill
from metrics.collection.XMoverScore import XMoverScore
from project_root import ROOT_DIR
from xai.Explainer import Explainer
from xai.evaluation.eval4nlp_evaluate import evaluate_mlqe_auc
from xai.util.corpus_explainer import explain_corpus


class XMSExplainer(Explainer):
    '''
    An explainer that uses feature importance scores directly extracted from XMoverScore. XMoverScore
    was proposed by
    Wei Zhao, Goran Glavaš, Maxime Peyrard, Yang Gao, Robert West, and Steffen Eger. “On the Lim-
    itations of Cross-lingual Encoders as Exposed by Reference-Free Machine Translation Evaluation”.
    In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online:
    Association for Computational Linguistics, July 2020, pp. 1656–1671. url: https://www.aclweb.
    org/anthology/2020.acl-main.151.
    '''

    def restructure(self, explanations, mode, cat, hyp_list):
        '''
        :param explanations: The output of XMoverscore (metrics/collection/metric_libs/xmoverscore*)
        :param mode: The mode(s) of XMS (CLP_1, UMD_1, CLP_2, UMD_2). Should be a list. CLP2 and UMD2 are not supported with
                     attributions
        :param cat: If multiple modes are selected this specifies concatenation via SUM or MIN
        :param hyp_list: A list of hypotheses
        :return:
        '''
        hyp_relevances = []
        src_relevances = []
        scores = []

        for x in range(len(hyp_list)):
            # I exposed the minimal distances of xms distance matrices in this value
            hyp_selection = []
            src_selection = []
            score_selection = []
            if 'CLP_1' in mode:
                src_selection.append(explanations[0][2][x])
                hyp_selection.append(explanations[0][1][x])
                score_selection.append(explanations[0][0][x])
            if 'UMD_1' in mode:
                src_selection.append(explanations[4][2][x])
                hyp_selection.append(explanations[4][1][x])
                score_selection.append(explanations[4][0][x])
            if 'CLP_2' in mode:
                src_selection.append(explanations[2][2][x])
                hyp_selection.append(explanations[2][1][x])
                score_selection.append(explanations[2][0][x])
            if 'UMD_2' in mode:
                src_selection.append(explanations[6][2][x])
                hyp_selection.append(explanations[6][1][x])
                score_selection.append(explanations[6][0][x])

            # I invert the scores, as they should be feature importances (there is another inversion in the evaluation)

            # Aggregate results of multiple modes
            if len(hyp_selection) > 1:
                hyp_relevance = list(zip(*hyp_selection))
                src_relevance = list(zip(*src_selection))
                if cat == 'SUM':
                    hyp_relevances.append([(attr[0][0], -sum([att[1] for att in attr])) for attr in hyp_relevance])
                    src_relevances.append([(attr[0][0], -sum([att[1] for att in attr])) for attr in src_relevance])

                elif cat == 'MIN':
                    hyp_relevances.append([(attr[0][0], -min([att[1] for att in attr])) for attr in hyp_relevance])
                    src_relevances.append([(attr[0][0], -min([att[1] for att in attr])) for attr in src_relevance])

            # for a single mode
            else:
                hyp_relevance = hyp_selection[0]
                hyp_relevances.append([(attr[0], -attr[1]) for attr in hyp_relevance])
                src_relevance = src_selection[0]
                src_relevances.append([(attr[0], -attr[1]) for attr in src_relevance])

            scores.append(sum(score_selection))
        return src_relevances, hyp_relevances, scores

    def apply_explanation(self, df, metrics=None, recover=False, xlm=False, drop_punctuation=True, models=None, embed=None, cat=None, k=2):
        '''
        :param df: dataframe with samples
        :param metrics: should be XMoverScore
        :param recover: Whether to recover previous explanations
        :param xlm: Whether to use XMS with XLMR or mBERT
        :param drop_punctuation: When using xlm=False, this specifies whether punctuation or subwords are dropped
        :param models: When using xlm=True, this specifies the (layer,model,suffix of the remapping file) to consider
        :param embed: Mapping mode, or a combinaion of multiple. See below for the possible mappings
        :param cat: The concatenation mode if multiple mappings are specified
        :param k: Embedding using remapping trained on k000 samples. Might not exist for all configurations
                  see /metrics/collection/metric_libs/xmoverscore and /metrics/collection/metric_libs/xmoverscore_xlmr
        :return: explanations in a dictionary format
        '''
        lp_list = df['LP'].tolist()
        ref_list = df['REF'].tolist()
        hyp_list = df['HYP'].tolist()
        src_list = df['SRC'].tolist()

        # Different modes of xms
        if embed == 'ALL':
            mode = ['CLP_1','UMD_1','CLP_2','UMD_2']
        elif embed == 'UNIGRAM':
            mode = ['CLP_1','UMD_1']
        elif embed == 'BIGRAM':
            mode = ['CLP_2','UMD_2']
        elif embed == 'CLP_1' or embed == 'CLP_2' or embed == 'UMD_1' or embed == 'UMD_2':
            mode = [embed]

        # Determine attributions and if possible ensemble them across remappings
        self.XMS = XMoverScore(layer=models[0][0], model_name=models[0][1], extension=models[0][2], xlm=xlm, drop_punctuation=drop_punctuation, k=k)
        explanations = self.XMS(src_list, hyp_list, lp_list[0], mode=mode, preprocess=False)
        src_relevance, hyp_relevance, scores = self.restructure(explanations, mode, cat, hyp_list)

        # Ensemble across multiple models
        models = models[1:]
        for model in models:
            self.XMS = XMoverScore(layer=model[0], model_name=model[1], extension=model[2], xlm=xlm, drop_punctuation=drop_punctuation, k=k)
            explanations = self.XMS(src_list, hyp_list, lp_list[0], mode=mode, preprocess=False)
            src_relevance2, hyp_relevance2, scores2 = self.restructure(explanations, mode, cat, hyp_list)

            hyp_relevance = [[(e_w[0],e_w[1] + e_w2[1]) for e_w, e_w2 in zip(e, e2)] for e, e2 in
                                zip(hyp_relevance, hyp_relevance2)]
            src_relevance = [[(e_w[0], e_w[1] + e_w2[1]) for e_w, e_w2 in zip(e, e2)] for e, e2 in
                         zip(src_relevance, src_relevance2)]

            scores = [(s + s2) for s, s2 in zip(scores, scores2)]


        # Save after every sample. This is saved as dill, as the explanations at this state can be various objects
        with open(os.path.join(ROOT_DIR,'xai/output/explanation_checkpoints/xms_backup.dill'), 'wb') as pickle_file:
            # Dill provides more options than pickle
            dill.dump(explanations, pickle_file, -1)


        def collapse_list_hash(l):
            # See BSExplainer.py
            words = [l[0][0]]
            scores = [l[0][1]]
            inb = 1
            for x in range(1, len(l)):
                if l[x][0][0:2] == '##':
                    words[-1] += l[x][0][2:]
                    scores[-1] += l[x][1]
                    inb += 1
                    if len(l)-1 == x:
                        scores[-1] /= inb
                else:
                    # Remove the underscore
                    scores[-1]/=inb
                    inb = 1
                    words.append(l[x][0])
                    scores.append(l[x][1])
            return list(zip(words, scores))

        def collapse_list(l):
            # See BSExplainer.py
            words = [l[0][0][1:]]
            scores = [l[0][1]]
            inb = 1
            for x in range(1, len(l)):
                if l[x][0][0] != '▁' and l[x][0][0] != 'Ġ':
                    words[-1] += l[x][0]
                    scores[-1] += l[x][1]
                    inb += 1
                    if len(l)-1 == x:
                        scores[-1] /= inb
                else:
                    # Remove the underscore
                    scores[-1]/=inb
                    inb = 1
                    words.append(l[x][0][1:])
                    scores.append(l[x][1])
            return list(zip(words, scores))

        # Remove begin and end tokens and collapse where necessary
        if '##' in ''.join([''.join([e[0] for e in src_relevance[x]])for x in range(len(src_relevance))]):
            src_relevance = [collapse_list_hash(e[1:-1]) for e in src_relevance]
        elif '▁' in ''.join([''.join([e[0] for e in src_relevance[x]])for x in range(len(src_relevance))]):
            src_relevance = [collapse_list(e[1:-1]) for e in src_relevance]
        if '##' in ''.join([''.join([e[0] for e in hyp_relevance[x]])for x in range(len(hyp_relevance))]):
            hyp_relevance = [collapse_list_hash(e[1:-1]) for e in hyp_relevance]
        elif '▁' in ''.join([''.join([e[0] for e in hyp_relevance[x]])for x in range(len(hyp_relevance))]):
            hyp_relevance = [collapse_list(e[1:-1]) for e in hyp_relevance]



        attributions = []
        for x in range(len(hyp_relevance)):
            attributions.append({'src': src_list[x], 'ref': ref_list[x], 'hyp': hyp_list[x],
                                 'metrics': {'XMOVERSCORE': {'attributions': hyp_relevance[x], 'src_attributions': src_relevance[x], 'score': scores[x]}}
                                 })

        return attributions


if __name__ == '__main__':
    # Example of using XLMR(Ensemlble) for the ro_en dev set of the Eval4NLP shared task
    XMSE = XMSExplainer()
    et_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_dev_et_en.tsv')
    ro_en_path = os.path.join(ROOT_DIR,'metrics/corpora/pandas_corpora/eval4nlp_dev_ro_en.tsv')
    n = 1000

    explain_corpus(XMSE,
                   recover=False,
                   from_row=0, to_row=n,
                   outfile='mlqe_et_attributions_xms_ens.json',
                   mlqe_pandas_path=ro_en_path,
                   #models=[(12,'','.map')],
                   #models=[(21,'xlm-roberta-large', '.map_base')],
                   models = [(16, 'joeddav/xlm-roberta-large-xnli', '.map_nli1'), (17, 'vicgalle/xlm-roberta-large-xnli-anli', '.map_nli2'), (17,'xlm-roberta-large', '.map_base')],
                   xlm = False, drop_punctuation=True,
                   embed ='UNIGRAM', cat='SUM', k='30k')

    evaluate_mlqe_auc([
        os.path.join(ROOT_DIR,'xai/output/explanations/0_'+str(n-1)+'_mlqe_et_attributions_xms_ens.json')],
        start=0, end=n, invert=True, f=100,
        corpus_path=ro_en_path)

    #explain_corpus(XMSE,
    #               recover=False,
    #               from_row=0, to_row=n,
    #               outfile='mlqe_ro_attributions_xms.json',
    #               mlqe_pandas_path=ro_en_path)

    #evaluate_mlqe_auc([
    #    os.path.join(ROOT_DIR,'xai/output/explanations/0_'+str(n-1)+'_mlqe_ro_attributions_xms.json')],
    #    start=0, end=n, invert=True, f=100,
    #    corpus_path=ro_en_path)