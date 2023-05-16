#
# SPICE model implementation to check dimensions
#

import torch
from torch import nn

print(torch.cuda.is_available())
x = torch.arange(0, 128, 1).type(torch.FloatTensor).reshape(1, 1, -1)
print('Input Size: ',x.size())


h = nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1 ,bias=True)  # same padding ,
mx = nn.MaxPool1d(kernel_size=3, stride=2, padding=1 )  # valid padding, stride=kernel_size
y = h(x)
print('Layer 1: ', y.size())
y=mx(y)
print('Layer 1 after Pooling: ', y.size())

  
h2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1 ,bias=True) 
g = h2(y)
print('Layer 2: ', g.size())
g=mx(g)
print('Layer 2 after Pooling: ', g.size())


h3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1 ,bias=True)
j = h3(g)
print('Layer 3: ', j.size())
j=mx(j)
print('Layer 3 after Pooling: ', j.size())


h4 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
kk = h4(j)  # since same
print('Layer 4: ', kk.size())
kk=mx(kk)
print('Layer 4 after Pooling: ', kk.size())



h5 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
jj = h5(kk)
print('Layer 5: ', jj.size())
jj=mx(jj)
print('Layer 5 after Pooling: ', jj.size())


h6 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
m = h6(jj)
print('Layer 6: ', m.size())
m=mx(m)
print('Layer 6 after Pooling: ', m.size())

print("After flattening", m.flatten().size())
''''''