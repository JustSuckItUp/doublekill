import torch
import models
from time import time

img = torch.randn(1, 3, 256, 256)
net = models.MobileViT_S()
img = img.cuda()
net = net.cuda()
net = net.eval()
for i in range(20):
    out = net(img)
tik = time()
for i in range(1000):
    out = net(img)
tok = time()
print((tok-tik)/1000)