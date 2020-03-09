import torch

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils.text.symbols import symbols
from utils.text import text_to_sequence
from utils.dsp import reconstruct_waveform
from utils import hparams as hp
import numpy as np


def init_hparams(hp_file):
    hp.configure(hp_file)


def get_forward_model(model_path):
    device = torch.device('cuda')
    model = ForwardTacotron(embed_dims=hp.forward_embed_dims,
                            num_chars=len(symbols),
                            durpred_rnn_dims=hp.forward_durpred_rnn_dims,
                            durpred_conv_dims=hp.forward_durpred_conv_dims,
                            rnn_dim=hp.forward_rnn_dims,
                            postnet_k=hp.forward_postnet_K,
                            postnet_dims=hp.forward_postnet_dims,
                            prenet_k=hp.forward_prenet_K,
                            prenet_dims=hp.forward_prenet_dims,
                            highways=hp.forward_num_highways,
                            n_mels=hp.num_mels).to(device)
    model.load(model_path)
    return model


def get_wavernn_model(model_path):
    device = torch.device('cuda')
    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode).to(device)

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
        voc_model.generate(m, '/tmp/', True, hp.voc_target, hp.voc_overlap, hp.mu_law)
    return wav

