import numpy as np
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

    with torch.no_grad():
        model.eval()
        embeddings = model(x)['embedding'].data.cpu().numpy()
        
    return embeddings


def get_timestamp_embeddings(x, model):

    with torch.no_grad():
        model.eval()
        sed_embeddings = model(x)['sed_embedding'].data.cpu().numpy()
        
    batch_size, frames_num, embedding_size = sed_embeddings.shape
    time_steps = np.arange(frames_num) * 0.01   # (frames_num,)
    time_steps = np.tile(time_steps, (batch_size, 1))   # (batch_size, frames_num)

    return sed_embeddings, time_steps
