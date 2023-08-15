import torch
from torch.autograd import Variable

N, D = 3,4

x = Variable(torch.randn(N,D), requires_grad=True)
y = Variable(torch.randn(N,D), requires_grad=True)
z = Variable(torch.randn(N,D), requires_grad=True)

# Run on GPU using .cuda()
# But it will not work here

# x = Variable(torch.randn(N,D).cuda(), requires_grad=True)
# y = Variable(torch.randn(N,D).cuda(), requires_grad=True)
# z = Variable(torch.randn(N,D).cuda(), requires_grad=True)


a = x*y
b = a+z
c = torch.sum(b)

c.backward()

print("X : ",x.grad.data)
print("Y : ",y.grad.data)
print("Z : ",z.grad.data)
