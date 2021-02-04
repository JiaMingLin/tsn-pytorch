import torch
weight_path = '~/.cache/torch/checkpoints/bn_inception-9f5701afb96c8044.pth'
weight_fixed_path = '~/.cache/torch/checkpoints/bn_inception-9f5701afb96c8044_fixed.pth'
state_dict = torch.load(weight_path)
for name, weights in state_dict.items():
    if 'bn' in name :
        state_dict[name] = weights.squeeze(0)
torch.save(state_dict,weight_fixed_path)