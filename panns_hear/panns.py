import torch

from .models import Cnn14


def load_model(model_file_path, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = Cnn14(
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

    model.sample_rate = 32000  # Input sample rate
    model.scene_embedding_size = 2048
    model.timestamp_embedding_size = 2048

    return model


def get_scene_embeddings(x, model):
    return model(x)['embedding']


def get_timestamp_embeddings(x, model):
    embedding = model(x)['embedding']
    embedding = embedding.unsqueeze(1).repeat(1, 248, 1)
    duration = x.shape[1] / model.sample_rate
    frame_duration = duration / embedding.shape[1]
    timestamps = torch.arange(frame_duration / 2, duration, frame_duration)
    timestamps = (timestamps * 1000).repeat(len(x), 1)
    return embedding, timestamps
