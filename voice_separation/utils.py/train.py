import os
import math
import torch
import torch.nn as nn
import traceback

# from .adabound import AdaBound
from .audio import Audio
from .evaluation import validate
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
# from model.embedder import ECAPAEmbedder


def train(device, args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str):
    best_sdr = -float("inf")  # initialize to a very low value
    # load embedder
    embedder_pt = torch.load(args.embedder_path)
    embedder = SpeechEmbedder(hp).to(device)
    embedder.load_state_dict(embedder_pt)
    embedder.eval()
    # embedder = ECAPAEmbedder().model.cuda().eval()  # Use this if you want to use ECAPA-TDNN as embedder

    audio = Audio(hp)
    model = VoiceFilter(hp).to(device)
    # if hp.train.optimizer == 'adabound':
    #     optimizer = AdaBound(model.parameters(),
    #                          lr=hp.train.adabound.initial,
    #                          final_lr=hp.train.adabound.final)
    if hp.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hp.train.adam)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.95,
            patience=7,       # Reduce LR after 7 “pseudo-epochs” (see below)
            min_lr=1e-7,
            verbose=True
        )

    else:
        raise Exception("%s optimizer not supported" % hp.train.optimizer)

    step = 0

    if chkpt_path is not None:
        logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path,weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
                                    
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.95,
        #     patience=8,
        #     min_lr=1e-7,
        #     verbose=True
        # )

        step = checkpoint['step']
        best_sdr = checkpoint.get("best_sdr", -float("inf"))

        # # Override learning rate
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = hp.train.adam

        # will use new given hparams.
        if hp_str != checkpoint['hp_str']:
            logger.warning("New hparams is different from checkpoint.")
    else:
        logger.info("Starting new training run")

    # if chkpt_path is not None:
    #     logger.info("Loading model weights from checkpoint: %s" % chkpt_path)
    #     checkpoint = torch.load(chkpt_path, weights_only=False)
    #     model.load_state_dict(checkpoint['model'])
    #     logger.info("Model weights loaded. Starting fresh optimizer and counters.")

    #     # Re-create optimizer and scheduler fresh
    #     optimizer = torch.optim.Adam(model.parameters(),
    #                                 lr=hp.train.adam)
                                    
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='max',
    #         factor=0.95,
    #         patience=7,
    #         min_lr=1e-7,
    #         verbose=True
    #     )

    #     # Reset step count and best SDR
    #     step = 0
    #     best_sdr = -float("inf")

    # else:
    #     logger.info("Starting new training run")

    try:
        criterion = nn.MSELoss()
        while True:
            model.train()
            for dvec_mels, target_mag, mixed_mag in trainloader:
                target_mag = target_mag.to(device)
                mixed_mag = mixed_mag.to(device)

                dvec_list = list()
                for mel in dvec_mels:
                    mel = mel.to(device)
                    dvec = embedder(mel)
                    dvec_list.append(dvec)
                dvec = torch.stack(dvec_list, dim=0)
                dvec = dvec.detach()

                mask = model(mixed_mag, dvec)
                output = mixed_mag * mask

                # Power compression
                # target_mag_p = torch.pow(target_mag + 1e-8, hp.audio.power)
                # output_mag_p = torch.pow(output + 1e-8, hp.audio.power)

                # Loss (MSE on compressed magnitudes)
                loss = criterion(output, target_mag)
                # loss = criterion(output, target_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step() 
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    logger.error("Loss exploded to %.02f at step %d!" % (loss, step))
                    raise Exception("Loss exploded")

                # write loss to tensorboard
                if step % hp.train.summary_interval == 0:
                    writer.log_training(loss, step)
                    logger.info("Wrote summary at step %d" % step)

                # 1. save checkpoint file to resume training
                # 2. evaluate and save sample to tensorboard
                if step % hp.train.checkpoint_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    sdr = validate(device, audio, model, embedder, testloader, writer, step, hp, lr)
                    scheduler.step(sdr)

                    if sdr > best_sdr:
                        best_sdr = sdr
                        save_path = os.path.join(pt_dir, 'best_model.pt')
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'step': step,
                            'hp_str': hp_str,
                            'best_sdr': best_sdr,
                        }, save_path)
                        logger.info(f"Saved new best model at step {step} with SDR {best_sdr:.2f} dB. Saved to {save_path}")
                
    except Exception as e:
        logger.info("Exiting due to exception: %s" % e)
        traceback.print_exc()
