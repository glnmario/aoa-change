import argparse
import json
import os
import warnings
import torch
import time
import logging
import itertools
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from gensim import utils as gensim_utils

logger = logging.getLogger(__name__)


class PathLineSentences(object):
    """Like :class:`~gensim.models.word2vec.LineSentence`, but process all files in a directory
    in alphabetical order by filename.

    The directory must only contain files that can be read by :class:`gensim.models.word2vec.LineSentence`:
    .bz2, .gz, and text files. Any file not ending with .bz2 or .gz is assumed to be a text file.

    The format of files (either text, or compressed text files) in the path is one sentence = one line,
    with words already preprocessed and separated by whitespace.

    Warnings
    --------
    Does **not recurse** into subdirectories.

    """
    def __init__(self, source, limit=None, max_sentence_length=100000):
        """
        Parameters
        ----------
        source : str
            Path to the directory.
        limit : int or None
            Read only the first `limit` lines from each file. Read all if limit is None (the default).

        """
        self.source = source
        self.limit = limit
        self.max_sentence_length = max_sentence_length

        if os.path.isfile(self.source):
            logger.debug('single file given as source, rather than a directory of files')
            logger.debug('consider using models.word2vec.LineSentence for a single file')
            self.input_files = [self.source]  # force code compatibility with list of files
        elif os.path.isdir(self.source):
            self.source = os.path.join(self.source, '')  # ensures os-specific slash at end of path
            logger.info('reading directory %s', self.source)
            self.input_files = os.listdir(self.source)
            self.input_files = [self.source + filename for filename in self.input_files]  # make full paths
            self.input_files.sort()  # makes sure it happens in filename order
        else:  # not a file or a directory, then we can't do anything with it
            raise ValueError('input is neither a file nor a path')
        logger.info('files read into PathLineSentences:%s', '\n'.join(self.input_files))

    def __iter__(self):
        """iterate through the files"""
        for file_name in self.input_files:
            logger.info('reading file %s', file_name)
            with gensim_utils.file_or_filename(file_name) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim_utils.to_unicode(line, encoding='utf-8').split()
                    i = 0
                    while i < len(line):
                        yield line[i:i + self.max_sentence_length]
                        i += self.max_sentence_length


def get_context(tokenizer, token_ids, target_position, sequence_length):
    window_size = int((sequence_length - 2) / 2)

    # determine where context starts and if there are any unused context positions to the left
    if target_position - window_size >= 0:
        start = target_position - window_size
        extra_left = 0
    else:
        start = 0
        extra_left = window_size - target_position

    # determine where context ends and if there are any unused context positions to the right
    if target_position + window_size + 1 <= len(token_ids):
        end = target_position + window_size + 1
        extra_right = 0
    else:
        end = len(token_ids)
        extra_right = target_position + window_size + 1 - len(token_ids)

    # redistribute to the left the unused right context positions
    if extra_right > 0 and extra_left == 0:
        if start - extra_right >= 0:
            padding = 0
            start -= extra_right
        else:
            padding = extra_right - start
            start = 0
    # redistribute to the right the unused left context positions
    elif extra_left > 0 and extra_right == 0:
        if end + extra_left <= len(token_ids):
            padding = 0
            end += extra_left
        else:
            padding = end + extra_left - len(token_ids)
            end = len(token_ids)
    else:
        padding = extra_left + extra_right

    context_ids = token_ids[start:end]
    context_ids = [tokenizer.cls_token_id] + context_ids + [tokenizer.sep_token_id]
    item = {'input_ids': context_ids + padding * [tokenizer.pad_token_id],
            'attention_mask': len(context_ids) * [1] + padding * [0]}

    new_target_position = target_position - start + 1

    return item, new_target_position


class ContextsDataset(torch.utils.data.Dataset):

    def __init__(self, i2w, sentences, context_size, tokenizer, sampling_probs, n_sentences=None):

        # todo:
        # keep a dictionary of how often a word has been sampled

        super(ContextsDataset).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.sampling_probs = sampling_probs

        for sentence in tqdm(sentences, total=n_sentences):
            token_ids = tokenizer.encode(' '.join(sentence), add_special_tokens=False)
            for spos, tok_id in enumerate(token_ids):
                if tok_id in i2w:
                    if np.random.random() <= self.sampling_probs[tok_id]:
                        word = i2w[tok_id]
                        model_input, pos_in_context = get_context(tokenizer, token_ids, spos, context_size)
                        self.data.append((model_input, word, pos_in_context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        model_input, lemma, pos_in_context = self.data[index]
        model_input = {'input_ids': torch.tensor(model_input['input_ids'], dtype=torch.long).unsqueeze(0),
                       'attention_mask': torch.tensor(model_input['attention_mask'], dtype=torch.long).unsqueeze(0)}
        return model_input, lemma, pos_in_context


def set_seed(seed, n_gpus):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus > 0:
        torch.cuda.manual_seed_all(seed)


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', '-m', type=str, required=True,
        help='Huggingface model name or directory path of a serialised model.')
    arg('--hdims', type=int, required=True,
        help="The model's hidden dimensions. These will be the dimensions of the contextualised embeddings.")
    arg('--data_path', '-d', type=str, required=True,
        help='Path to corpus or corpus directory (iterates through files).')
    arg('--targets_path', '-t', type=str, required=True,
        help='Path to json file containing the list of target words.')
    arg('--output_path', '-o', type=str, required=True,
        help='Output path for the contextualised embeddings.')
    arg('--context_size', '-c', type=int, default=512,
        help="The context window size around a target word occurrence.")
    arg('--batch_size', '-b', type=int, default=64,
        help="The batch size at inference time.")
    arg('--max_occurrences', '-x', type=int, default=None,
        help="The batch size at inference time.")
    arg('--local_rank', type=int, default=-1,
        help="For distributed runs.")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.info(__file__.upper())
    start_time = time.time()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        n_gpu = 1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        n_gpu,
        bool(args.local_rank != -1)
    )

    # Set seeds across modules
    set_seed(42, n_gpu)

    # Load targets
    with open(args.targets_path, 'r') as f:
        targets = [w for w in json.load(f) if type(w) == str]

    logger.warning('N targets: {}'.format(len(targets)))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, never_split=targets)
    model = AutoModelForMaskedLM.from_pretrained(args.model, output_hidden_states=True)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    # Store vocabulary indices of target words
    unk_id = tokenizer.convert_tokens_to_ids('[UNK]')
    targets_ids = [tokenizer.encode(t, add_special_tokens=False) for t in targets]
    assert len(targets) == len(targets_ids)
    i2w = {}
    for t, t_id in tqdm(zip(targets, targets_ids)):
        if len(t_id) > 1 or (len(t_id) == 1 and t_id == unk_id):
            # logger.warning('{} not in vocabulary!'.format(t))
            continue
        else:
            i2w[t_id[0]] = t

    logger.warning('{}/{} words in vocabulary.'.format(len(i2w), len(targets)))

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Get sentence iterator
    sentences = PathLineSentences(args.data_path)

    # with warnings.catch_warnings():
    #     warnings.resetwarnings()
    #     warnings.simplefilter("always")
    nSentences = 0
    target_counter = {target: 0 for target in i2w}
    for sentence in sentences:
        nSentences += 1
        for tok_id in tokenizer.encode(' '.join(sentence), add_special_tokens=False):
            if tok_id in target_counter:
                target_counter[tok_id] += 1

    logger.warning('Total usages: %d' % (sum(list(target_counter.values()))))

    sampling_probs = {target: 1. for target in i2w}
    if args.max_occurrences:
        for w, fr in target_counter.items():
            if fr > args.max_occurrences:
                sampling_probs[w] = args.max_occurrences / target_counter[w]
                target_counter[w] = args.max_occurrences
            else:
                target_counter[w] = fr

        logger.warning('Subsampled usages: %d' % (sum(list(target_counter.values()))))

    # Container for usages
    usages = {
        i2w[target]: np.empty((target_count, args.hdims))  # usage matrix
        for (target, target_count) in target_counter.items()
    }

    # Iterate over sentences and collect representations
    nUsages = 0
    curr_idx = {i2w[target]: 0 for target in target_counter}

    def collate(batch):
        return [
            {'input_ids': torch.cat([item[0]['input_ids'] for item in batch], dim=0).squeeze(1),
             'attention_mask': torch.cat([item[0]['attention_mask'] for item in batch], dim=0).squeeze(1)},
            [item[1] for item in batch],
            [item[2] for item in batch]
        ]

    dataset = ContextsDataset(i2w, sentences, args.context_size, tokenizer, sampling_probs, nSentences)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
    iterator = tqdm(dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

    for step, batch in enumerate(iterator):
        model.eval()
        batch_tuple = tuple()
        for t in batch:
            try:
                batch_tuple += (t.to(device),)
            except AttributeError:
                batch_tuple += (t,)

        batch_input_ids = batch_tuple[0] #.squeeze(1)
        batch_lemmas, batch_spos = batch_tuple[1], batch_tuple[2]

        #logger.warning(batch_input_ids['input_ids'].shape)
        #logger.warning(len(batch_lemmas))
        #logger.warning(len(batch_spos))

        with torch.no_grad():
            if torch.cuda.is_available():
                batch_input_ids['input_ids'] = batch_input_ids['input_ids'].to('cuda')
                batch_input_ids['attention_mask'] = batch_input_ids['attention_mask'].to('cuda')

            outputs = model(**batch_input_ids)
            #logger.warning(type(outputs))
            #logger.warning(len(outputs))

            if torch.cuda.is_available():
                hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[1]]
            else:
                hidden_states = [l.clone().numpy() for l in outputs[1]]

            # store usage tuples in a dictionary: lemma -> (vector, position)
            for b_id in np.arange(batch_input_ids['input_ids'].shape[0]):
                lemma = batch_lemmas[b_id]

                # layers = [layer[b_id, batch_spos[b_id] + 1, :] for layer in hidden_states]
                #usage_vector = np.concatenate(layers)
                # usage_vector = layers[-1]
                usage_vector = hidden_states[-1][b_id, batch_spos[b_id] + 1, :]
                usages[lemma][curr_idx[lemma], :] = usage_vector

                curr_idx[lemma] += 1
                nUsages += 1

    for lemma in usages:
        logger.warning(type(usages[lemma]), usages[lemma].shape)

    iterator.close()
    logger.warning(usages)
    np.savez_compressed(**usages, file=args.output_path)

    logger.warning('usages: %d' % (nUsages))
    logger.warning("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
