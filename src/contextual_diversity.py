import argparse
import json
import string
from tqdm import tqdm

def main():
    """
    Contextual Diversity - number of types that co-occur with a target word in the same sentence.
    """
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input_file', '-i', type=str,
        help='Path to target text file.', required=True)
    arg('--output_path', '-o', type=str,
        help='Path to output file (tab-separated, .tsv).', required=True)
    arg('--target_words_path', '-t', type=str,
        help='Path to target words file.', required=True)
    arg('--window_size', '-w', type=int,
        help='Size of one side of the co-occurrence window. The actual window size will be twice this value.',
        default=5)
    # arg('--discard_punctuation', '-d',
    #     help='Whether to discard punctuation.',
    #     action='store_true')
    args = parser.parse_args()

    with open(args.target_words_path, 'r') as f:
        targets = [w for w in json.load(f) if type(w) == str]

    with open(args.input_file, 'r') as f:
        lines = f.readlines()

    # Dictionary mapping each target word to the set of contexts with which it co-occurs
    contexts = {w: set() for w in targets}

    for line in tqdm(lines):
        print(line)
        tokens_tmp = line.split()
        tokens = []
        for token in tokens_tmp:
            if not all(c in string.punctuation for c in token):
                tokens.append(token.lower())
        print(tokens)
        print()
        for i, token in enumerate(tokens):
            if token in contexts:
                left_window = tokens[max(0, i - args.window_size): i]
                right_window = tokens[i + 1:i + args.window_size + 1]
                contexts[token].update(left_window + right_window)

    with open(args.output_path, 'w') as f:
        for target in targets:
            f.write('{}\t{}\n'.format(target, len(contexts[target])))

if __name__ == '__main__':
    main()
