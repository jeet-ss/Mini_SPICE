#
# SPICE model implementation to check dimensions
#

import torch
from torch import nn

batch_size = 10


print("Encoder-----------------------------------")
x = torch.arange(0, 1280, 1).type(torch.FloatTensor).reshape(batch_size, 1, -1)
print('Input Size: ',x.size())


c1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, stride=1 ,bias=True)  # same padding ,
mx = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, return_indices=True )  # valid padding, stride=kernel_size
x = c1(x)
print('Layer 1: ', x.size())
x, i1 = mx(x)
print('Layer 1 after Pooling: ', x.size())
  
c2 = nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1 ,bias=True) 
x = c2(x)
print('Layer 2: ', x.size())
x, i2 = mx(x)
print('Layer 2 after Pooling: ', x.size())


c3 = nn.Conv1d(128, 256, kernel_size=3, padding=1, stride=1 ,bias=True)
x = c3(x)
print('Layer 3: ', x.size())
x, i3 = mx(x)
print('Layer 3 after Pooling: ', x.size(), i3.size())


c4 = nn.Conv1d(256, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
x = c4(x)  # since same
print('Layer 4: ', x.size())
x, i4 = mx(x)
print('Layer 4 after Pooling: ', x.size(), i4.size())



c5 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
x = c5(x)
print('Layer 5: ', x.size())
x, i5 = mx(x)
print('Layer 5 after Pooling: ', x.size(), i5.size())


c6 = nn.Conv1d(512, 512, kernel_size=3, padding=1, stride=1 ,bias=True)
x = c6(x)
print('Layer 6: ', x.size())
x, i6 = mx(x)
print('Layer 6 after Pooling: ', x.size(), i6.size())

x=x.reshape(batch_size, -1)
print("After flattening", x.size())
f1 = nn.Linear(1024, 48)
f2 = nn.Linear(48, 1)
out = f2(f1(x))
print("output of Encoder: ", out.size() )
''''''

print("Decoder-------------------------------")
inp = out.reshape(batch_size,1,-1)
print("input size: ", inp.size())

fb1 = nn.Linear(1, 48)
fb2 = nn.Linear(48, 1024)
inp = (fb1(inp)).reshape(batch_size, -1, 48)
print("after fc layers: ", inp.size())

channel_dec_list = [1, 256, 256, 256, 128, 64, 64]
#channel_dec_list = [512, 512, 512, 256, 128, 64, 1]  # for reverse duplicate


mp = nn.MaxUnpool1d(kernel_size=3, stride=2, padding=1)
#
if inp.size() == i6.size():
    pass
    #inp = mp(inp, i6, output_size = (batch_size, 512, 4))
    #print('Layer 1 after UnPooling: ', inp.size())
t1 = nn.ConvTranspose1d(channel_dec_list[0], channel_dec_list[1], kernel_size=3, padding=1, stride=1, bias=True)

inp = t1(inp)
print('Layer 1: ', inp.size())

#imp = torch.floor(i5[:,:256,:]/2).to(torch.int64)  # take only the first 256 channels
#print('inp', imp.size())
if inp.size() == i5.size():
    inp = mp(inp, i5, output_size=(batch_size, 512, 8))
    print('Layer 2 after UnPooling: ', inp.size())
t2 = nn.ConvTranspose1d(channel_dec_list[1], channel_dec_list[2], kernel_size=3, padding=1, stride=1, bias=True)
inp = t2(inp)
print('Layer 2: ', inp.size())
inp, i = mx(inp)
print('Layer 2 after pool: ', inp.size())

imp2 = torch.floor(i4[:,:256,:6]/2).to(torch.int64)  # take only the first 256 channels
#print('inp', imp2.size())
if inp.size() == i4.size():
    inp = mp(inp, i4, output_size=(batch_size, 512, 16))
    print('Layer 3 after UnPooling: ', inp.size())
t3 = nn.ConvTranspose1d(channel_dec_list[2], channel_dec_list[3], kernel_size=3, padding=1, stride=1, bias=True)
inp = t3(inp)
print('Layer 3: ', inp.size())
inp,i = mx(inp)
print('Layer 3 after pool: ', inp.size())

if inp.size() == i3.size():
    inp = mp(inp, i3, output_size=(batch_size, 256, 32))
    print('Layer 4 after UnPooling: ', inp.size())
t4 = nn.ConvTranspose1d(channel_dec_list[3], channel_dec_list[4], kernel_size=3, padding=1, stride=1, bias=True)
inp = t4(inp)
print('Layer 4: ', inp.size())
inp, i = mx(inp)
print('Layer 4 after pooling: ', inp.size())


if inp.size() == i2.size():
    inp = mp(inp, i2, output_size=(batch_size, 128, 64))
    print('Layer 5 after UnPooling: ', inp.size())
t5 = nn.ConvTranspose1d(channel_dec_list[4], channel_dec_list[5], kernel_size=3, padding=1, stride=1, bias=True)
inp = t5(inp)
print('Layer 5: ', inp.size())
inp,i  = mx(inp)
print('Layer 5 after pool: ', inp.size())
xx, i = mx(inp)
print(xx.size())

if inp.size() == i1.size():
    inp = mp(inp, i1, output_size=(batch_size, 1, 128))
    print('Layer 6 after UnPooling: ', inp.size())
t6 = nn.ConvTranspose1d(channel_dec_list[5], channel_dec_list[6], kernel_size=3, padding=1, stride=1, bias=True)
inp = t6(inp)
print('Layer 6: ', inp.size())
inp, i = mx(inp)
print('Layer 6 after pool: ', inp.size())