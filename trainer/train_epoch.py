from tqdm import tqdm
import torch


def train_epoch(model, vocoder, optimizer, scheduler, dataloader, criterion, featurizer, aligner, logger, epoch,
                melspec_config, config
                ):
    model.train()
    for i, batch in tqdm(enumerate(next(iter(dataloader)))):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)
        with torch.no_grad():
            batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

        batch.durations *= (batch.waveform_length // melspec_config.hop_length + 1).unsqueeze(-1)

        optimizer.zero_grad()
        output = model(batch)
        losses = criterion(spect, output[1], batch.durations, output[2])
        loss = losses[0] + losses[1]
        loss.backward()
        optimizer.step()
        # log all loses
        logger.set_setup(i + epoch * len(dataloader))
        logger.add_scalar('spect_loss', losses[0].item())
        logger.add_scalar('duration_loss', losses[1].item())
        logger.add_scalar('combined loss', loss.item())

        # we use only one batch
        break
    scheduler.step()
    # return losses[0], losses[1]
