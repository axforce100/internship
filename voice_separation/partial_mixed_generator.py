import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
import soundfile as sf

from utils.audio import Audio
from utils.hparams import HParam


def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))


def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix_dual(hp, args, audio, base_num, s1_dvec, s1_target, s2_dvec, s2_target, train=True):
    # choose output directory based on train flag
    subdir = 'train3' if train else 'test_unseen_partial_ol'
    dir_ = os.path.join(args.out_dir, subdir)
    os.makedirs(dir_, exist_ok=True)  # ensure subdir exists

    srate = hp.audio.sample_rate

    # Load and trim all audio
    paths = [s1_dvec, s1_target, s2_dvec, s2_target]
    audios = [librosa.load(p, sr=srate)[0] for p in paths]
    audios = [librosa.effects.trim(a, top_db=20)[0] for a in audios]
    d1, w1, d2, w2 = audios

    for w in audios:
        assert len(w.shape) == 1, 'wav files must be mono'

    # Ensure d-vector reference is long enough
    if d1.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return "skipped"
    if d2.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return "skipped"

    # Apply VAD if requested
    if args.vad == 1:
        w1, w2 = vad_merge(w1), vad_merge(w2)

    # Clip to length
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return "skipped"
    w1, w2 = w1[:L], w2[:L]

        # ====================== NEW MIXING ======================
    L = int(srate * hp.data.audio_len)

    # mixture length will be based on speaker1 (full length)
    mix_len = L

    # speaker1 occupies entire audio
    mix_w1 = np.zeros(mix_len, dtype=np.float32)
    mix_w1[0:L] = w1[:L]

    # speaker2 occupies only 50% to 100%
    start = int(0.5 * L)
    end   = int(1 * L)

    mix_w2 = np.zeros(mix_len, dtype=np.float32)
    mix_w2[start:end] = w2[:end-start]  # only use as much as needed

    mixed = mix_w1 + mix_w2

    # normalize
    norm = np.max(np.abs(mixed)) * 1.1
    if norm == 0:
        return "skipped"
    mix_w1 /= norm
    mix_w2 /= norm
    mixed /= norm
    # ========================================================

    # Save BOTH copies of mixed wav
    for i in range(2):
        sample_idx = base_num * 2 + i
        mixed_wav_path = formatter(dir_, hp.form.mixed.wav, sample_idx)
        mixed_mag_path = formatter(dir_, hp.form.mixed.mag, sample_idx)
        sf.write(mixed_wav_path, mixed, srate)
        mixed_mag, _ = audio.wav2spec(mixed)
        torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # Save target speaker 1 (with offset applied waveform)
    target_wav_path_1 = formatter(dir_, hp.form.target.wav, base_num * 2 + 0)
    target_mag_path_1 = formatter(dir_, hp.form.target.mag, base_num * 2 + 0)
    dvec_txt_path_1 = formatter(dir_, hp.form.dvec, base_num * 2 + 0)
    sf.write(target_wav_path_1, mix_w1, srate)
    target_mag_1, _ = audio.wav2spec(mix_w1)
    torch.save(torch.from_numpy(target_mag_1), target_mag_path_1)
    with open(dvec_txt_path_1, 'w') as f:
        f.write(s1_dvec)

    # Save target speaker 2 (with offset applied waveform)
    target_wav_path_2 = formatter(dir_, hp.form.target.wav, base_num * 2 + 1)
    target_mag_path_2 = formatter(dir_, hp.form.target.mag, base_num * 2 + 1)
    dvec_txt_path_2 = formatter(dir_, hp.form.dvec, base_num * 2 + 1)
    sf.write(target_wav_path_2, mix_w2, srate)
    target_mag_2, _ = audio.wav2spec(mix_w2)
    torch.save(torch.from_numpy(target_mag_2), target_mag_path_2)
    with open(dvec_txt_path_2, 'w') as f:
        f.write(s2_dvec)

    return "ok"

def robust_train_generate(hp, args, audio, spk_list, total_samples=50000, train=True):
    assert total_samples % 2 == 0, "total_samples must be even"
    total_mixtures = total_samples // 2
    base_num = 0
    trials = 0
    max_trials = 100000
    pbar = tqdm.tqdm(total=total_mixtures)
    while base_num < total_mixtures and trials < max_trials:
        trials += 1
        try:
            spk1, spk2 = random.sample(spk_list, 2)
            s1_dvec, s1_target = random.sample(spk1, 2)
            s2_dvec, s2_target = random.sample(spk2, 2)
            result = mix_dual(hp, args, audio, base_num, s1_dvec, s1_target, s2_dvec, s2_target, train=train)
            if result != "skipped":
                base_num += 1
                pbar.update(1)
        except Exception as e:
            print(f"[Skip] Trial {trials}, Error: {e}")
            continue
    if trials == max_trials:
        print("Warning: Max trials reached. Some mixtures may not be generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-d', '--libri_dir', type=str, default=None)
    parser.add_argument('-v', '--voxceleb_dir', type=str, default=None)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('--vad', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'], help='whether to generate train or test mixtures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train3'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test_unseen'), exist_ok=True)

    hp = HParam(args.config)
    audio = Audio(hp)

    if args.libri_dir is None and args.voxceleb_dir is None:
        raise Exception("Please provide directory of data")

    if args.libri_dir is not None:
        train_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'train/train-clean-100', '*')) if os.path.isdir(x)] + \
                        [x for x in glob.glob(os.path.join(args.libri_dir, 'train/train-clean-360', '*')) if os.path.isdir(x)]
        test_folders = [x for x in glob.glob(os.path.join(args.libri_dir, 'dev-clean', '*'))]

    train_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True) for spk in train_folders]
    train_spk = [x for x in train_spk if len(x) >= 2]

    test_spk = [glob.glob(os.path.join(spk, '**', hp.form.input), recursive=True) for spk in test_folders]
    test_spk = [x for x in test_spk if len(x) >= 2]

    print("Found %d speakers in training set, %d speakers in test set." % (len(train_spk), len(test_spk)))

    if args.mode == 'train':
        robust_train_generate(hp, args, audio, train_spk, total_samples=50000, train=True)
    else:
        robust_train_generate(hp, args, audio, test_spk, total_samples=100, train=False)
