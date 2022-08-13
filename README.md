# transformers-fsd50k
HUBERT/wav2vec2 pretrained on FSD50K

Setup on lambdalabs A100

Later could be docker

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y pybind11-dev
sudo apt-get install -y python3-pybind11
sudo apt-get install -y mlocate

# https://github.com/facebookresearch/fairseq

# Can't get APEX working, pytorch b0rks
#git clone https://github.com/NVIDIA/apex
#cd apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
#  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
#  --global-option="--fast_multihead_attn" ./

cd
sudo pip3 install soundfile
sudo pip3 install fairseq
sudo pip3 install tensorboardX

#git clone https://github.com/facebookresearch/fairseq.git
#cd fairseq
#sudo pip3 install --editable ./
#sudo pip3 install --editable ./


# Get a dataset of synth audio that has normalized hz and length
wget https://zenodo.org/record/4677097/files/surge-velocity64-2K.tar
tar xvf surge-velocity64-2K.tar

export FAIRSEQ=/usr/local/lib/python3.8/dist-packages/fairseq

python $FAIRSEQ/examples/wav2vec/wav2vec_manifest.py surge/ --dest surge.manifest --ext ogg --valid-percent  0.01

# Should actually train/valid stratify by patch. Anyway.

export PYTHONPATH=$PYTHONPATH:$FAIRSEQ

# Needed for pyaudio
#sudo pip3 install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# Oups that instal sm80
# so try
#sudo pip3 install --upgrade --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu116

#sudo apt install -y nvtop


# Edit  $FAIRSEQ/examples/wav2vec/config/pretraining/wav2vec2_base_librispeech.yaml
# or $FAIRSEQ/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml
for distributed_world_size: 1
# Or the number of GPUs you have
not 64
in task: add
  sample_rate: 44100
# Might want to "adjust the encoder architecture to keep 100hz rate"
# https://github.com/facebookresearch/fairseq/issues/1280
and in common: add
  tensorboard_logdir: tb
  wandb_project: wav2vec2-surge-pitch

# Maybe also play with
#  batch_size: 4
#  num_workers: 6

fairseq-hydra-train task.data=/home/ubuntu/surge.manifest/ --config-dir $FAIRSEQ/examples/wav2vec/config/pretraining --config-name wav2vec2_base_librispeech
#Or 
fairseq-hydra-train task.data=/home/ubuntu/surge.manifest/ --config-dir $FAIRSEQ/examples/wav2vec/config/pretraining --config-name wav2vec2_large_librivox


# Here are librispeech instructions
https://cloud.google.com/tpu/docs/tutorials/wav2vec2-pytorch


sudo pip3 install wandb

