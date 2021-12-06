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

torch.manual_seed(57)
torch.cuda.manual_seed(57)
torch.cuda.manual_seed_all(57)
np.random.seed(57)
torch.backends.cudnn.deterministic = True

root = '../'


# https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer

class CustomScheduler:
    def __init__(self, model_size, optimizer, warmup, factor=2):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._step = 0

    def rate(self, step):
        return 1 / self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._step += 1
        rate = self.rate(self._step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()


def train(model, vocoder, dataloader, test_texts, scheduler, criterion, featurizer, aligner, tknz, logger,
          melspec_config, config):
    vocoder.eval()
    for epoch in range(config.n_epochs):
        print(f'Start of the epoch {epoch}')
        train_epoch(model, vocoder, scheduler, dataloader, criterion, featurizer, aligner, tknz, logger, epoch,
                    melspec_config, config)
        # if (epoch + 1) % config.show_every == 0:
        #     valid(model, vocoder, test_texts, criterion, featurizer, aligner, tknz, logger, epoch, melspec_config,
        #           config)


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
    featurizer = MelSpectrogram(melspec_config).to(config.device)
    aligner = GraphemeAligner().to(config.device)

    # data
    dataset = LJSpeechDataset(root=root)
    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True, batch_size=config.batch_size,
                            collate_fn=LJSpeechCollator()
                            )

    # model
    model = FastSpeechModel(config.dict_size, config.emb_dim, config.n_blocks1, config.n_blocks2, config.n_heads,
                            config.conv_size, config.kernel_size, config.duration_size, config.attn_size,
                            config.output_size, config.device, config.dropout, config.alpha
                            ).to(config.device)
    vocoder = Vocoder().to(config.device)

    # optmizations
    criterion = CustomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CustomScheduler(config.emb_dim, optimizer, config.warmup)

    # wandb
    logger = WanDBWriter(config)

    tknz = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    # training
    train(model, vocoder, dataloader, test_texts, scheduler, criterion, featurizer, aligner, tknz, logger,
          melspec_config, config)
