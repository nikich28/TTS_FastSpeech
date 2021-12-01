import torch
from tqdm import tqdm


@torch.no_grad()
def valid(model, dataloader, criterion, featurizer, aligner, logger, epoch, melspec_config, config):
    model.eval()
    for i, batch in tqdm(enumerate(next(iter(dataloader)))):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)
        batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

        batch.durations *= (batch.waveform_length // melspec_config.hop_length + 1).unsqueeze(-1)

        output = model(batch)
        losses = criterion(spect, output[1], batch.durations, output[2])
        loss = losses[0] + losses[1]

        # log all loses
        logger.set_setup(i + epoch * len(dataloader), mode='valid')
        logger.add_scalar('spect_loss', losses[0].item())
        logger.add_scalar('duration_loss', losses[1].item())
        logger.add_scalar('combined loss', loss.item())

        break

    # return losses[0], losses[1]
