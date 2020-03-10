# ⏩ ForwardTacotron

Inspired by Microsoft's [FastSpeech](https://www.microsoft.com/en-us/research/blog/fastspeech-new-text-to-speech-model-improves-on-speed-accuracy-and-controllability/)
we modified Tacotron (Fork from fatchord's [WaveRNN](https://github.com/fatchord/WaveRNN)) to generate speech in a single forward pass using a duration predictor to align text and generated mel spectrograms. Hence, we call the model ForwardTacotron (see Fig. 1).

<p align="center">
  <img src="assets/model.png" width=0.8/>
</p>
<p align="center">
  <b>Figure 1:</b> Model Architecture.
</p>

The model has following advantages:
- **Robustness:** No repeats and failed attention modes for challenging sentences.
- **Speed:** The generation of a mel spectogram takes about 0.04s on a GeForce RTX 2080.
- **Controllability:** It is possible to control the speed of the generated utterance.
- **Efficiency:** In contrast to FastSpeech and Tacotron, the model of ForwardTacotron
does not use any attention. Hence, the required memory grows linearly with text size, which makes it possible to synthesize large articles at once.


## 🔈 Samples

[Can be found here.](https://as-ideas.github.io/ForwardTacotron/)

The samples are generated with a model trained 100K steps on LJSpeech together with the pretrained WaveRNN vocoder 
provided by the WaveRNN repo. Both models are commited in the pretrained folder. You can try them out with the following notebook:  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/ForwardTacotron/blob/master/notebooks/synthesize.ipynb)

## ⚙️ Installation

Make sure you have:

* Python >= 3.6
* [PyTorch 1 with CUDA](https://pytorch.org/)

Then install the rest with pip:
```
pip install -r requirements.txt
```

## 🚀 Training your own Model

(1) Download and preprocess the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset:
 ```
python preprocess.py --path /path/to/ljspeech
```
(2) Train Tacotron with:
```
python train_tacotron.py
```
(3) Use the trained tacotron model to create alignment features with:
```
python train_tacotron.py --force_align
```
(4) Train ForwardTacotron with:
```
python train_forward.py
```
(5) Generate Sentences with Griffin-Lim vocoder:
```
python gen_forward.py --alpha 1 --input_text "this is whatever you want it to be" griffinlim
```
As in the original repo you can also use a trained WaveRNN vocoder:
```
python gen_forward.py --input_text "this is whatever you want it to be" wavernn
```
____

## References

* [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)

## Acknowlegements

* [https://github.com/keithito/tacotron](https://github.com/keithito/tacotron)
* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
* [https://github.com/xcmyz/LightSpeech](https://github.com/xcmyz/LightSpeech)

## Maintainers

* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)

## Copyright

See [LICENSE](LICENSE) for details.
