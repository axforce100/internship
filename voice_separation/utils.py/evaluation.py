# import torch
# import torch.nn as nn
# from mir_eval.separation import bss_eval_sources


# def validate(audio, model, embedder, testloader, writer, step, hp):
#     model.eval()
    
#     criterion = nn.MSELoss()
#     total_loss = 0.0
#     total_sdr = 0.0
#     num_samples = 0

#     with torch.no_grad():
#         for batch in testloader:
#             dvec_mel, target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase = batch[0]

#             dvec_mel = dvec_mel.cuda()
#             target_mag = target_mag.unsqueeze(0).cuda()
#             mixed_mag = mixed_mag.unsqueeze(0).cuda()

#             dvec = embedder(dvec_mel)
#             dvec = dvec.unsqueeze(0)
#             est_mask = model(mixed_mag, dvec)
#             est_mag = est_mask * mixed_mag

#             target_mag_p = torch.pow(target_mag + 1e-8, hp.audio.power)
#             est_mag_p = torch.pow(est_mag + 1e-8, hp.audio.power)

#             test_loss = criterion(target_mag_p, est_mag_p).item()
#             # test_loss = criterion(target_mag, est_mag).item()

#             mixed_mag = mixed_mag[0].cpu().detach().numpy()
#             target_mag = target_mag[0].cpu().detach().numpy()
#             est_mag = est_mag[0].cpu().detach().numpy()
#             est_wav = audio.spec2wav(est_mag, mixed_phase)
#             est_mask = est_mask[0].cpu().detach().numpy()

#             sdr = bss_eval_sources(target_wav, est_wav, False)[0][0]
#             if num_samples == 0 :
#                 writer.log_evaluation(test_loss, sdr,
#                                     mixed_wav, target_wav, est_wav,
#                                     mixed_mag.T, target_mag.T, est_mag.T, est_mask.T,
#                                     step)
                
            
#             total_loss += test_loss
#             total_sdr += sdr
#             num_samples += 1

#     avg_loss = total_loss / num_samples
#     avg_sdr = total_sdr / num_samples
#     writer.log_validation_metrics(avg_loss, avg_sdr, step)
#     print(f"[Validation] Step {step}: Avg Loss = {avg_loss:.4f}, Avg SDR = {avg_sdr:.2f} dB")
#     model.train()


import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources


def validate(device, audio, model, embedder, testloader, writer, step, hp, lr):
    model.eval()
    
    total_loss = 0.0
    total_sdr = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in testloader:
            # Unpack batch of individual samples
            dvec_list, target_wav_list, mixed_wav_list, target_mag_batch, mixed_mag_batch, mixed_phase_list = zip(*batch)
            batch_size = len(dvec_list)

            # Stack spectrograms to [B, F, T]
            target_mag_batch = torch.stack(target_mag_batch).to(device)
            mixed_mag_batch = torch.stack(mixed_mag_batch).to(device)

            # Compute d-vectors (assuming embedder accepts [1, F, T] input per sample)
            dvec_batch = [embedder(dvec.to(device)) for dvec in dvec_list]
            dvec_batch = torch.stack(dvec_batch).to(device)# [B, D]

            # Forward pass
            est_mask_batch = model(mixed_mag_batch, dvec_batch)  # [B, F, T]
            est_mag_batch = est_mask_batch * mixed_mag_batch     # [B, F, T]

            # # Apply power compression
            # target_mag_p = torch.pow(target_mag_batch + 1e-8, hp.audio.power)
            # est_mag_p = torch.pow(est_mag_batch + 1e-8, hp.audio.power)

            # Correct loss: per-sample loss, then sum and normalize by total samples
            loss_per_sample = torch.mean((est_mag_batch - target_mag_batch) ** 2, dim=(1, 2))  # shape: [B]
            batch_loss = loss_per_sample.sum().item()
            total_loss += batch_loss

            # Compute SDR per sample
            for i in range(batch_size):
                est_mag_np = est_mag_batch[i].cpu().numpy()
                target_mag_np = target_mag_batch[i].cpu().numpy()
                est_wav = audio.spec2wav(est_mag_np, mixed_phase_list[i])
                sdr = bss_eval_sources(target_wav_list[i], est_wav, False)[0][0]

                total_sdr += sdr
                num_samples += 1

                # Log the first sample of the first batch
                if num_samples == 1:
                    writer.log_evaluation(
                        #loss_per_sample[i].item(), sdr,
                        mixed_wav_list[i], target_wav_list[i], est_wav,
                        mixed_mag_batch[i].cpu().numpy().T,
                        target_mag_np.T,
                        est_mag_np.T,
                        est_mask_batch[i].cpu().numpy().T,
                        step
                    )

    # Compute average loss and SDR per sample
    avg_loss = total_loss / num_samples
    avg_sdr = total_sdr / num_samples

    writer.log_validation_metrics(avg_loss, avg_sdr, lr, step)
    print(f"[Validation] Step {step}: Avg Loss = {avg_loss:.4f}, Avg SDR = {avg_sdr:.2f} dB")
    model.train()
    return avg_sdr  # Return values for potential further use