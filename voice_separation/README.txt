Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking (CNN + LSTM)
==============================================================================================
Installation / Requirements
Python 3.12 with PyTorch 2.6.0

pip install requirements.txt

===============================================================================================

Raw Dataset
 http://www.openslr.org/12/ (LibriSpeech)
 Train:
 -train-clear-100.tar.gz (252 speakers)
 -train-clear-360.tar.gz (922 speakers)
 Test:
 -dev-clean.tar.gz (40 speakers)

 ==========================================================================================================

Data Generation 

run python voice_separation/generator2.py -c [config yaml]  -d [raw data directory] -o [output directory]

Example:
python voice_separation/generator2.py -c voice_separation/config/default.yaml  -d LibriSpeech -o voice_separation/data

===========================================================================================================

Train Model

run python trainer.py -c [config yaml] -e [path of embedder pt file] -m [name]

Example:
python voice_separation/trainer.py -c voice_separation/config/default.yaml -e ./Models/embedder.pt -m experiment_n

========================================================================================================================================

Resume Model Training from a checkpoint_path

run python trainer.py -c [config yaml] --checkpoint_path [chkpt/name/chkpt_{step}.pt] -e [path of embedder pt file] -m name

Example:
python voice_separation/trainer.py -c voice_separation/config/default.yaml --checkpoint_path checkpoints/experiment_n/best_model.pt -e ./Models/embedder.pt -m experiment_n

============================================================================================================================================

View Training progress

tensorboard --logdir ./logs

=====================================================================================================================================

inference

run python inference.py -c [config yaml] -e [path of embedder pt file] --checkpoint_path [path of chkpt pt file] -m [path of mixed wav file] -r [path of reference wav file] -o [output directory]

Example:
python voice_separation/inference.py -c voice_separation/config/default.yaml -e Models/embedder.pt --checkpoint_path checkpoints/experiment_01/best_model.pt -m voice_separation/data/train3/000868-mixed.wav  -r  LibriSpeech/train/train-clean-360\598\127703\598-127703-0033-norm.wav -o voice_separation/output
 -o voice_separation/output

 =================================================================================================


 Main Files
 
 -dataloader.py: defines the dataset and dataloader pipeline for model Training -utils/audio.py: audio processing utilities
 -generator2.py: mized audio data generation 
 -inference.py: Runs inference with model to separate target speech from mixtures.
 -config/default.yaml: experiment hyperparameters, paths, and settings
 -utils/evaluation.py: model validation code (Calculation of SDR & loss for test dataset)
 -utils/plotting.py: spectrogram plotting for tensorboard
 -utils/train.py: Main code for model training loop
 -utils/writer.py: custom logging class (MyWriter) that extends SummaryWriter from TensorBoardX to track training progress and evaluation results

=========================================================

Models
 -checkpoints/experiment_12/best_model.pt (Main Model)
 -Models/embedder.pt (Speaker Encoder)



=======================================================================================================================================================