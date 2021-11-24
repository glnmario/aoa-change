import json
import pickle
import numpy as np
from docopt import docopt
import logging
from scipy.spatial.distance import pdist
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Average pairwise distance (APD) algorithm


def mean_pairwise_distance(word_usages, metric):
    """
    Computes the mean pairwise distance between two usage matrices.

    :param word_usages: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param metric: a distance metric compatible with `scipy.spatial.distance.cdist`
    (e.g. 'cosine', 'euclidean')
    :return: the mean pairwise distance between two usage matrices
    """
    if isinstance(word_usages, tuple):
        usage_matrix, _, _ = word_usages
    else:
        usage_matrix = word_usages

    if usage_matrix.shape[0] == 0:
        logger.info('In T1: {}'.format(usage_matrix.shape[0] > 0))
        return 0.

    return np.mean(pdist(usage_matrix, metric=metric))


def mean_variance(word_usages):
    """
    Computes the mean pairwise distance between two usage matrices.

    :param word_usages: a three-place tuple including, in this order, a usage matrix, a list of
    snippets, and a list of integers indicating the lemma's position in the snippet
    :param metric: a distance metric compatible with `scipy.spatial.distance.cdist`
    (e.g. 'cosine', 'euclidean')
    :return: the mean pairwise distance between two usage matrices
    """
    if isinstance(word_usages, tuple):
        usage_matrix, _, _ = word_usages
    else:
        usage_matrix = word_usages

    if usage_matrix.shape[0] == 0:
        logger.info('In T1: {}'.format(usage_matrix.shape[0] > 0))
        return 0.

    return np.mean(np.var(usage_matrix), axis=0)


def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute synchronic coherence among contextualised representations.

    Usage:
        distance.py [--metric=<d> --frequency] <testSet> <valueFile> <outPath>

    Arguments:
        <testSet> = path to file with one target per line
        <valueFile> = path to file containing usage matrices and snippets
        <outPath> = output path for result file

    Options:
        --metric=<d>  The distance metric, which must be compatible with
        `scipy.spatial.distance.cdist` [default: cosine]
        --frequency    Output frequency as well.

    Note:
        Assumes pickled dictionaries as input:
        {t: (usage_matrix, snippet_list, target_pos_list) for t in targets}

    """)

    testset = args['<testSet>']
    value_file = args['<valueFile>']
    outpath = args['<outPath>']
    distmetric = args['--metric']

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
        raise ValueError('valueFile: wrong format.')

    # Print only targets to output file
    with open(outpath, 'w', encoding='utf-8') as f_out:
        for target in tqdm(targets):
            try:
                distance = mean_pairwise_distance(usages[target], distmetric)
                variance = mean_variance(usages[target])
                logger.warning(target)
            except KeyError:
                continue

            if args['--frequency']:
                frequency = usages[target].shape[0]
                f_out.write('{}\t{}\t{}\n'.format(target, distance, variance, frequency))
            else:
                f_out.write('{}\t{}\n'.format(target, distance, variance))

    # logging.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
