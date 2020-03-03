import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from pathlib import Path


def get_tts_datasets(path: Path, batch_size, r):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []

    for (item_id, len) in dataset:
        if len <= hp.tts_max_mel_len:
            dataset_ids += [item_id]
            mel_lengths += [len]

    with open(path/'text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    train_dataset = TTSDataset(path, dataset_ids, text_dict)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           num_workers=1,
                           pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))

    # Used to evaluate attention during training process
    attn_example = dataset_ids[longest]

    # print(attn_example)
    return train_set, attn_example


class TTSDataset(Dataset):

    def __init__(self, path: Path, dataset_ids, text_dict, alignments=False):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict
        self.alignments = alignments

    def __getitem__(self, index):
        item_id = self.metadata[index]
        x = text_to_sequence(self.text_dict[item_id], hp.tts_cleaner_names)
        mel = np.load(self.path/'mel'/f'{item_id}.npy')
        mel_len = mel.shape[-1]
        if self.alignments:
            alg = np.load(self.path/'alg'/f'{item_id}.npy')
        else:
            alg = np.zeros((mel.shape[0], 1))
        return x, mel, item_id, mel_len, alg

    def __len__(self):
        return len(self.metadata)


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    alg = [pad1d(x[4][:max_x_len], max_x_len) for x in batch]
    alg = np.stack(alg)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    alg = torch.tensor(alg)

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, ids, mel_lens, alg

