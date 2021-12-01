from tqdm import tqdm
import torch


def train_epoch(model, vocoder, optimizer, scheduler, dataloader, criterion, featurizer, aligner, logger, epoch,
                melspec_config, config
                ):
    model.train()
    for i, batch in tqdm(enumerate(dataloader)):
        batch = batch.to(config.device)

        with torch.no_grad():
            durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)
        spect = featurizer(batch.waveform)

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

        if (epoch + 1) % config.show_every == 0:
            with torch.no_grad():
                smp = output[0][0].unsqueeze(0)
                sr = melspec_config.sr
                preds = vocoder.inference(smp)
                logger.add_audio('Generated_audio', preds, sample_rate=sr)
                logger.add_audio('Real_audio', batch.waveform[0], sample_rate=sr)
                logger.add_text('Text', batch.transcript[0])

        # we use only one batch
        break
    scheduler.step()
    # return losses[0], losses[1]
