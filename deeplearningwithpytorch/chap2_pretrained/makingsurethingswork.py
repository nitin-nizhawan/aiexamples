import torch

print(torch.version.__version__)

a = torch.ones(3,3)
b = torch.ones(3,3)

print(a+b)


a = a.to("cuda")
b = b.to("cuda")

print(a + b)

