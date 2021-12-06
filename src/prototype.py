import json
import pickle
import numpy as np
from docopt import docopt
import logging
from scipy.spatial.distance import cdist
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Average pairwise distance (APD) algorithm


def mean_pairwise_distance(word_usages1, word_usages2, metric):
    """
    Computes the mean pairwise distance between two usage matrices.

    :param word_usages1: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param word_usages2: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param metric: a distance metric compatible with `scipy.spatial.distance.cdist`
    (e.g. 'cosine', 'euclidean')
    :return: the mean pairwise distance between two usage matrices
    """
    if isinstance(word_usages1, tuple):
        usage_matrix1, _, _ = word_usages1
    else:
        usage_matrix1 = word_usages1

    if isinstance(word_usages2, tuple):
        usage_matrix2, _, _ = word_usages2
    else:
        usage_matrix2 = word_usages2

    if usage_matrix1.shape[0] == 0 or usage_matrix2.shape[0] == 0:
        logger.info('In T1: {}   In T2: {}'.format(usage_matrix1.shape[0] > 0, usage_matrix2.shape[0] > 0))
        return 0.

    return np.mean(cdist(usage_matrix1, usage_matrix2, metric=metric))


def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute (diachronic) distance between sets of contextualised representations.

    Usage:
        prototype.py <testSet> <valueFile> <outPath>

    Arguments:
        <testSet> = path to file with one target per line
        <valueFile> = path to file containing usage matrices and snippets
        <outPath> = output path for result file

    Note:
        Assumes pickled dictionaries as input:
        {t: (usage_matrix, snippet_list, target_pos_list) for t in targets}

    """)

    testset = args['<testSet>']
    value_file = args['<valueFile>']
    outpath = args['<outPath>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info(__file__.upper())
    # start_time = time.time()

    # Load targets
    with open(testset, 'r') as f:
        targets = [w for w in json.load(f) if type(w) == str]

    # Get usages collected from corpus 1
    if value_file.endswith('.dict'):
        with open(value_file, 'rb') as f_in:
            usages = pickle.load(f_in)
    elif value_file.endswith('.npz'):
        usages = np.load(value_file)
    else:
        raise ValueError('valueFile 1: wrong format.')


    # Print only targets to output file
    n_ = 0
    prototype_vectors = {}

    for target in tqdm(targets):
        try:
            prototype_vectors[target] = np.mean(usages[target], axis=0)
        except KeyError:
            continue

        logger.info(target)
        n_ += 1
    logger.info('{} target words in this time period.'.format(n_))

    ndims = prototype_vectors[list(prototype_vectors.keys())[0]].shape[-1]

    with open(outpath, 'w', encoding='utf-8') as f_out:
        f_out.write('{} {}\n'.format(n_, ndims))
        for w in prototype_vectors:
            f_out.write('{} {}\n'.format(
                w,
                ' '.join(map(str, prototype_vectors[w]))
            ))

    # logging.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
