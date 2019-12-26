import torch
import torch.nn as nn

# input = torch.randn(1, 64, 14, 128, 128).cuda(1)
# conv1 = nn.Conv3d(64,128,kernel_size=(3, 3, 3), stride=(1,1,1), padding=(1,1,1)).cuda(1)
# output = conv1(input)
# targert = output.new(*output.size())
# targert.data.uniform_(-0.01, 0.01)
# error = (targert - output).mean()
# error.backward()
# print(output.shape)

import math
import random

for i in range(0,100):
    a = random.randint(1500,2700)

    b = 3
    c = 4
    d = 5


    # b = random.randint(3,5)
    # c = random.randint(6,8)
    # d = random.randint(7,8)


    temp1 = math.floor(math.floor(math.floor(a/b)/c)/d)
    temp2 = math.floor(a/(b*c*d))

    if temp1 != temp2:
        print("something wrong")
