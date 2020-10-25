
from .corpora import *


CORPUS_LIST = {
    'EMVIC2014': EMVIC2014,
    'Cerf2007-FIFA': Cerf2007_FIFA,
    'ETRA2019': ETRA2019,
    # 'MIT-LowRes': MIT_LowRes,
}


frequencies = {
    1000: ['EMVIC2014', 'Cerf2007-FIFA'],
    500: ['ETRA2019'],  # 'IRCCyN_IVC_Eyetracker_Images_LIVE_Database'],
    # the ones from MIT are 240Hz but oh well
    250: ['MIT-LearningToPredict', 'MIT-LowRes', 'MIT-CVCL']  # 'Dorr2010-GazeCom',
}

def get_corpora(args, additional_corpus=None):
    if args.task:
        return {corpus: CORPUS_LIST[corpus](args)
                for corpus in TASK_CORPORA[args.task]}

    corpora = []
    for f, c in frequencies.items():
        if args.hz <= f:
            corpora.extend(c)

    # corpora = list(CORPUS_LIST.keys())

    # used to add a corpus to evaluator to test for overfitting during
    # training time
    # if isinstance(additional_corpus, str) and additional_corpus not in corpora:
    #     corpora.append(additional_corpus)
    #     logging.info('[evaluator] Added an unseen data set: {}'.format(
    #         additional_corpus))
    return {c: CORPUS_LIST[c](args) for c in corpora}
