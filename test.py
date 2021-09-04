import torch

import panns_hear


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = panns_hear.load_model('Cnn14_DecisionLevelMax_mAP=0.385.pth', device)

x = torch.Tensor(32, 480000).to(device)
z_s = panns_hear.get_scene_embeddings(x, model)
z_t = panns_hear.get_timestamp_embeddings(x, model)
print(z_s.shape)
print(z_t[0].shape)
print(z_t[1].shape)

assert(z_s.shape == (32, 2048))
assert(z_t[0].shape == (32, 248, 2048))
assert(z_t[1].shape == (32, 248))
print('Test passed')