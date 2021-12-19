import json
import pickle
import numpy as np
from docopt import docopt
import logging
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    """
    Compute (diachronic) distance between sets of contextualised representations.
    """

    # Get the arguments
    args = docopt("""Compute average contextualised representations.

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

    n_ = 0
    prototype_vectors = {}

    for target in tqdm(targets):
        try:
            prototype = np.mean(usages[target], axis=0)
        except KeyError:
            continue

        if not np.isnan(np.sum(prototype)):
            prototype_vectors[target] = np.mean(usages[target], axis=0)

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
