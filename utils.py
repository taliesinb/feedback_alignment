import torch

from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch import nn

class flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def MLP(*args, linearity=nn.Linear, nonlinearity=nn.Tanh):
	last = args[0]
	assert(isinstance(last, int))
	layers = []
	needs_non_linearity = False
	for spec in args[1:]:
		if isinstance(spec, int):
			if needs_non_linearity:
				layers.append(nonlinearity())
			needs_non_linearity = True
			layer = linearity(last, spec)
			last = spec
		else:
			layer = spec
		layers.append(layer)
	return nn.Sequential(*layers)

def toT(x):
    return torch.tensor(x, dtype=torch.float32, requires_grad=False)

def fromT(x):
    return x.detach().numpy()

def tensor_data(x, y, batch_size=64):
	dataset = TensorDataset(toT(x), toT(y))
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	return loader

def mnist_data(batch_size=64, is_train=True):
	ts = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	data = datasets.MNIST('../data', train=is_train, download=True, transform=ts)
	return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)