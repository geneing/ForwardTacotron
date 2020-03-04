import pickle
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from pathlib import Path
import random

def get_tts_datasets(path: Path, batch_size, r, alignments=False):

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

    train_dataset = TTSDataset(path, dataset_ids, text_dict, alignments=alignments)

    sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=sampler,
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
            dur = np.load(self.path/'alg'/f'{item_id}.npy')
        else:
            # dummy durations to simplify collate func
            dur = np.zeros((mel.shape[0], 1))
        return x, mel, item_id, mel_len, dur

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

    dur = [pad1d(x[4][:max_x_len], max_x_len) for x in batch]
    dur = np.stack(dur)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    dur = torch.tensor(dur).float()

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, ids, mel_lens, dur


class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)
