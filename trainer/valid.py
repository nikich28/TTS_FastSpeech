import torch
from tqdm import tqdm


@torch.no_grad()
def valid(model, dataloader, criterion, featurizer, aligner, logger, epoch, melspec_config, config):
    model.eval()
    for i, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)
        durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)

        durations = durations * (batch.waveform_length // melspec_config.hop_length + 1).unsqueeze(-1)
        batch.durations = durations.to(config.device)

        output = model(batch)
        losses = criterion(spect, output[0], batch.durations, output[1])
        loss = losses[0] + losses[1]

        # log all loses
        logger.set_step(i + epoch * len(dataloader), mode='valid')
        logger.add_scalar('spect_loss', losses[0].item())
        logger.add_scalar('duration_loss', losses[1].item())
        logger.add_scalar('combined loss', loss.item())

        break

    # return losses[0], losses[1]
