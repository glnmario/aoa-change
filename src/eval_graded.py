import argparse

import numpy as np
from docopt import docopt
from scipy.stats import spearmanr
import os


def get_ys(model_answers, true_answers):
    """
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: a numpy array for the model scores, and one for the true scores
    """
    y_hat = {}
    with open(model_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            try:
                lemma, score, _ = line.strip().split('\t')
            except ValueError:
                lemma, score, _, _ = line.strip().split('\t')
            if score == 'nan':
                continue
            y_hat[lemma] = float(score)

    y = {}
    with open(true_answers, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            lemma, score = line.strip().split('\t')
            y[lemma] = float(score)

    return y_hat, y


def eval_task2(model_answers, true_answers):
    """
    Computes the Spearman's correlation coefficient against the true rank as annotated by humans
    :param model_answers: path to tab-separated answer file (lemma + "\t" + score)
    :param true_answers: path to tab-separated gold answer file (lemma + "\t" + score)
    :return: (Spearman's correlation coefficient, p-value)
    """
    y_hat, y = get_ys(model_answers, true_answers)

    y_hat_ = []
    y_ = []
    cnt = 0
    for w in y:
        if w in y_hat:
            y_.append(y[w])
            y_hat_.append(y_hat[w])
            cnt += 1

    print('Correlation between {}/{} words'.format(cnt, len(y)))

    r, p = spearmanr(y_hat_, y_)
    return r, p


def main():
    """
    Evaluate lexical semantic change detection results.
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--gold_scores', '-g',
        help='Path to tab-separated answer file for Task 2 (lemma + "\t" + corr. coeff.)', required=True)
    arg('--predicted_scores', '-p',
        help='Path to tab-separated gold answer file for Task 2 (lemma + "\t" + corr. coeff.)', required=True)
    args = parser.parse_args()

    if os.path.isfile(args.predicted_scores):
        r, p = eval_task2(args.predicted_scores, args.gold_scores)
        print('Task 2 r: {:.3f}  p: {:.4f}'.format(r, p))
    else:
        print('Task 2 predictions not found!')


if __name__ == '__main__':
    main()