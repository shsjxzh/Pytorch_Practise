import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# unsqueeze 在这里是将一个一维张量变成了一个二维张量，在指定维度插入了一个1
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand_like(x)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.input = nn.Linear(n_feature, n_hidden)
        
        self.hidden_1 = nn.Linear(n_hidden, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.hidden_3 = nn.Linear(n_hidden, n_hidden)
        self.hidden_4 = nn.Linear(n_hidden, n_hidden)
        self.hidden_5 = nn.Linear(n_hidden, n_hidden)
        self.hidden_6 = nn.Linear(n_hidden, n_hidden)
        self.hidden_7 = nn.Linear(n_hidden, n_hidden)
        self.hidden_8 = nn.Linear(n_hidden, n_hidden)
        
        self.output = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = F.relu(self.hidden_6(x))
        x = F.relu(self.hidden_7(x))
        x = F.relu(self.hidden_8(x))
        
        x = self.output(x)
        return x
    
net = Net(n_feature = 1, n_hidden = 10, n_output = 1)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.2)  # lr 是学习率的意思
loss_func = nn.MSELoss() #指定损失函数（评价标准）为mean square error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

y = y.to(device)

for i in range(10000):
    x = x.to(device)
    prediction = net(x)
    loss = loss_func(prediction, y)
    
    optimizer.zero_grad() #清空上一次的梯度
    loss.backward()
    optimizer.step()
    
x, y, prediction = x.cpu(), y.cpu(), prediction.cpu()

plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 2)
print(loss)