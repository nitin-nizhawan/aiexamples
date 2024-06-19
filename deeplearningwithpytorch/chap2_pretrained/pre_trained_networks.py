from torchvision import models
import torch

dir(models)

print(models)

alexnet = models.AlexNet()


print(alexnet)

# 101 layer residual nn
resnet = models.resnet101(pretrained=True)


print(resnet)

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])




from PIL import Image
img = Image.open("../dlwpt-code/data/p1ch2/bobby.jpg")


img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)


## prepares restnet to put it into inference mode
resnet.eval()

## run rest net on our image
out = resnet(batch_t)


with open('../dlwpt-code/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)



percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100


print(labels[index[0]], percentage[index[0]].item())


_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])





