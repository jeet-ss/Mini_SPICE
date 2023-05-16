import torch
from torch import nn

print(torch.cuda.is_available())
x = torch.arange(0, 1024, 1).type(torch.FloatTensor).reshape(1, 1, -1)
print(x.size())
h = nn.Conv1d(1, 1024, kernel_size=512, padding=256, stride=4 ,bias=True)  # same padding ,
mx = nn.MaxPool1d(kernel_size=2, stride=None, padding=0 )  # valid padding, stride=kernel_size

y = h(x)
print(y.size())
y=mx(y.reshape(1, 1024, -1))
print(y.size())

kernels = [512, 64, 64, 64, 64, 64]
filters = 32*[32, 4, 4, 4, 8, 16]  # full model
strides = [4,1,1,1,1,1]


h2 = nn.Conv1d(1024, 128, kernel_size=64, padding=32, stride=1 ,bias=True) 
g = h2(y)
print(g.size())
g=mx(g)
print(g.size())


h3 = nn.Conv1d(128, 128, kernel_size=64, padding=32, stride=1 ,bias=True)
j = h3(g)
print(j.size())
j=mx(j)
print(j.size())


h4 = nn.Conv1d(128, 128, kernel_size=64, padding=32, stride=1 ,bias=True)
kk = h3(j)  # since same
print(kk.size())
kk=mx(kk)
print(kk.size())



h5 = nn.Conv1d(128, 256, kernel_size=64, padding=32, stride=1 ,bias=True)
jj = h5(kk)
print(jj.size())
jj=mx(jj)
print(jj.size())



h6 = nn.Conv1d(256, 512, kernel_size=64, padding=32, stride=1 ,bias=True)
m = h6(jj)
print(m.size())
m=mx(m)
print(m.size())
