import torch
import torchaudio

from model.FastSpeech import FastSpeechModel
from configs.config import TaskConfig
from loss.loss import CustomLoss
from model.Vocoder import Vocoder
from dataset.LJSpeechDataset import LJSpeechDataset
from dataset.DatasetCollator import LJSpeechCollator
from utils.aligner import GraphemeAligner
from utils.featurizer import MelSpectrogramConfig, MelSpectrogram
import numpy as np
from trainer.train_epoch import train_epoch
from trainer.valid import valid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from logger.logger import WanDBWriter
from tqdm import tqdm

torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)
np.random.seed(57)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # create config
    config = TaskConfig()

    test_texts = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
    ]

    # create utils
    melspec_config = MelSpectrogramConfig()

    # model
    model = FastSpeechModel(config.dict_size, config.emb_dim, config.n_blocks1, config.n_blocks2, config.n_heads,
                            config.conv_size, config.kernel_size, config.duration_size, config.attn_size,
                            config.output_size, config.device, config.dropout, config.alpha
                            ).to(config.device)
    model.load_state_dict(torch.load("best_model80.pth"))
    vocoder = Vocoder().to(config.device)

    # wandb
    logger = WanDBWriter(config)

    tknz = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    for i, text in tqdm(enumerate(test_texts)):
        logger.set_step(i, mode='valid')
        tokens, lens = tknz(text)
        tokens = tokens.to(config.device)
        spect = model(tokens, True)[0]
        wav = vocoder.inference(spect)
        sr = melspec_config.sr
        logger.add_audio(f'Generated_audio{i}', wav, sample_rate=sr)
