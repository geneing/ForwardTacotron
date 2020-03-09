import torch

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils.text.symbols import symbols
from utils.text import text_to_sequence
from utils.dsp import reconstruct_waveform
import numpy as np


def get_forward_model(model_path,):
    device = torch.device('cuda')
    model = ForwardTacotron(embed_dims=256,
                            num_chars=len(symbols),
                            durpred_rnn_dims=64,
                            durpred_conv_dims=256,
                            rnn_dim=512,
                            postnet_k=8,
                            postnet_dims=256,
                            prenet_k=16,
                            prenet_dims=256,
                            highways=4,
                            n_mels=80).to(device)
    model.load(model_path)
    return model


def get_wavernn_model(model_path):
    device = torch.device('cuda')
    model = WaveRNN(rnn_dims=512,
                    fc_dims=512,
                    bits=9,
                    pad=2,
                    upsample_factors=(5, 5, 11),
                    feat_dims=80,
                    compute_dims=128,
                    res_out_dims=128,
                    res_blocks=10,
                    hop_length=275,
                    sample_rate=22050,
                    mode='MOL').to(device)

    model.load(model_path)
    return model


def synthesize(input_text, tts_model, voc_model, alpha=1.0):
    x = text_to_sequence(input_text.strip(), ['english_cleaners'])
    m = tts_model.generate(x, alpha=alpha)
    # Fix mel spectrogram scaling to be from 0 to 1
    m = (m + 4) / 8
    np.clip(m, 0, 1, out=m)
    wav = None
    if voc_model == 'griffinlim':
        wav = reconstruct_waveform(m, n_iter=32)
    else:
        m = torch.tensor(m).unsqueeze(0)
        voc_model.generate(m, '', True, 11_000, 550, True)
    return wav
