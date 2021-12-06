from tqdm import tqdm
import torch
from .valid import valid


def train_epoch(model, vocoder, scheduler, dataloader, criterion, featurizer, aligner, tknz, logger, epoch,
                melspec_config, config
                ):
    model.train()
    for i, batch in tqdm(enumerate(dataloader), position=0, leave=True):
        batch = batch.to(config.device)

        spect = featurizer(batch.waveform)
        with torch.no_grad():
            durations = aligner(batch.waveform, batch.waveform_length, batch.transcript)

        durations = durations * (batch.waveform_length // melspec_config.hop_length + 1).unsqueeze(-1)
        batch.durations = durations.to(config.device)

        scheduler.zero_grad()
        output = model(batch)

        log_durs = torch.log(batch.durations + batch.durations.eq(0).float())

        losses = criterion(spect, output[0], log_durs, output[1])
        loss = losses[0] + losses[1]
        loss.backward()
        scheduler.step()
        # log all loses
        logger.set_step(i + epoch * len(dataloader))
        logger.add_scalar('spect_loss', losses[0].item())
        logger.add_scalar('duration_loss', losses[1].item())
        logger.add_scalar('combined loss', loss.item())

    if (epoch + 1) % config.show_every == 0:
        torch.save(model.state_dict(), f"best_model_{epoch + 1}.pth")
