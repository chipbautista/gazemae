
from .corpora import *


CORPUS_LIST = {
    'EMVIC2014': EMVIC2014,
    'Cerf2007-FIFA': Cerf2007_FIFA,
    'ETRA2019': ETRA2019,
    'MIT-LowRes': MIT_LowRes,
}


frequencies = {
    1000: ['EMVIC2014', 'Cerf2007-FIFA'],
    500: ['ETRA2019'],
    # the ones from MIT are 240Hz but oh well
    250: ['MIT-LowRes'],
}

def get_corpora(args, additional_corpus=None):
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
