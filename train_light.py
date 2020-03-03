import torch
import traceback
from torch import optim, nn
import torch.nn.functional as F

from models.fast_speech_cbhg import FastSpeechCBHG
from models.fast_speech_cbhg_lstm import FastSpeechCbhgLstm
from models.fast_speech_conv import FastSpeechConv
from models.fast_speech_gru import FastSpeechGru
from models.fast_speech import FastSpeech
from models.fast_speech_lstm import FastSpeechLstm
from models.fast_speech_lstm_cbhg import FastSpeechLstmCbhg
from models.fast_speech_lstm_post_old import FastSpeechLstmPost

from models.light_tts import LightTTS
from notebooks.utils.display import save_wav
from utils import hparams as hp
from utils.display import *
from utils.dataset import get_tts_datasets
from utils.dur_dataset import get_dur_datasets
from utils.text import sequence_to_text
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
import argparse
from utils import data_parallel_workaround
import os
from pathlib import Path
import time
import numpy as np
import sys
from utils.checkpoints import save_checkpoint, restore_checkpoint


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Light TTS Model
    print('\nInitialising Light TTS Model...\n')
    model = LightTTS(embed_dims=hp.tts_embed_dims,
                     num_chars=len(symbols),
                     n_mels=hp.num_mels,
                     postnet_dims=hp.tts_postnet_dims,
                     prenet_k=hp.tts_encoder_K,
                     lstm_dims=hp.tts_lstm_dims,
                     postnet_k=hp.tts_postnet_K,
                     num_highways=hp.tts_num_highways).to(device)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('fft', paths, model, optimizer, create_if_missing=True)

    if not force_gta:
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()

            r, lr, max_step, batch_size = session

            training_steps = max_step - current_step

            simple_table([(f'Steps with r={r}', str(training_steps//1000) + 'k Steps'),
                          ('Batch Size', batch_size),
                          ('Learning Rate', lr)])

            train_set, attn_example = get_dur_datasets(paths.data, batch_size, r)
            train_loop(paths, model, optimizer, train_set, lr, training_steps, attn_example)

    train_set, attn_example = get_dur_datasets(paths.data, 8, 1)
    create_gta_features(model, train_set, paths.gta)
    print('Training Complete.')


def train_loop(paths: Paths, model, optimizer, train_set, lr, train_steps, attn_example):
    device = next(model.parameters()).device  # use same device as model parameters

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1

    for e in range(1, epochs+1):

        start = time.time()
        running_loss = 0
        dur_running_loss = 0

        # Perform 1 epoch
        for i, (x, m, ids, mel_len, dur) in enumerate(train_set, 1):

            x, m, dur = x.to(device), m.to(device), dur.to(device)
            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                m_hat, post, dur_hat = data_parallel_workaround(model, x, m, dur)
            else:
                m_hat, post, dur_hat = model(x, m, dur)

            loss = F.l1_loss(m_hat, m)
            dur_loss = F.l1_loss(dur_hat, dur)
            loss = loss + dur_loss
            optimizer.zero_grad()

            loss.backward()

            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i
            dur_running_loss += dur_loss.item()
            dur_avg_loss = dur_running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'fast_speech_step{k}K'
                save_checkpoint('fft', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in ids and e > 1:
                idx = ids.index(attn_example)
                try:
                    seq = x[idx].tolist()
                    m_gen = model.generate(seq)
                    save_spectrogram(m_gen, paths.light_mel_plot / f'{step}_gen', 600)
                except Exception:
                    traceback.print_exc()
                save_spectrogram(np_now(m_hat[idx]), paths.light_mel_plot/f'{step}_gta', 600)
                save_spectrogram(np_now(m[idx]), paths.light_mel_plot/f'{step}_target', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} ' \
                  f'| Dur Loss: {dur_avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)
        model.log(paths.light_log, msg)

        save_checkpoint('light', paths, model, optimizer, is_silent=True)

        #print(' ')


def create_gta_features(model: Tacotron, train_set, save_path: Path):
    device = next(model.parameters()).device  # use same device as model parameters

    iters = len(train_set)

    for i, (x, mels, ids, mel_lens, dur) in enumerate(train_set, 1):

        x, mels, dur = x.to(device), mels.to(device), dur.to(device)

        with torch.no_grad(): gta, _, _ = model(x, mels, dur)

        gta = gta.cpu().numpy()

        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            mel = (mel + 4) / 8
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == "__main__":
    main()
