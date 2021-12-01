from tqdm import tqdm
import torch


def train_epoch(model, vocoder, optimizer, scheduler, dataloader, criterion, featurizer, aligner, logger, epoch,
                melspec_config, config
                ):
    model.train()
    for i, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)
        with torch.no_grad():
            durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)

        durations = durations * (batch.waveform_length // melspec_config.hop_length + 1).unsqueeze(-1)
        batch.durations = durations.to(config.device)

        optimizer.zero_grad()
        output = model(batch)
        losses = criterion(spect, output[0], batch.durations, output[1])
        loss = losses[0] + losses[1]
        loss.backward()
        optimizer.step()
        # log all loses
        logger.set_step(i + epoch * len(dataloader))
        logger.add_scalar('spect_loss', losses[0].item())
        logger.add_scalar('duration_loss', losses[1].item())
        logger.add_scalar('combined loss', loss.item())

        # we use only one batch
        break
    scheduler.step()
    # return losses[0], losses[1]
