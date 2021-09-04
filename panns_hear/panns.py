import torch
import torch.nn as nn

from torchaudio.transforms import Resample

from . import pytorch_utils
from .models import Cnn14_DecisionLevelMax


def load_model(model_file_path, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = Cnn14_DecisionLevelMax(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    model.to(device)

    # Set model weights using checkpoint file
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Add resampler to be compatible with input waveforms
    resampler = Resample(48000, 32000).to(device)
    model = nn.Sequential(resampler, model)

    model.sample_rate = 48000  # Input sample rate
    model.scene_embedding_size = 2048
    model.timestamp_embedding_size = 2048

    return model


def get_scene_embeddings(x, model):
    return model(x)['embedding'].mean(dim=1)


def get_timestamp_embeddings(x, model):
    embedding = model(x)['embedding']
    embedding = pytorch_utils.interpolate(embedding, ratio=8)
    duration = x.shape[1] / model.sample_rate
    frame_duration = duration / embedding.shape[1]
    timestamps = torch.arange(frame_duration / 2, duration, frame_duration)
    timestamps = (timestamps * 1000).repeat(len(x), 1)
    return embedding, timestamps
