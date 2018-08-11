import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 生成一系列的数据
n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)                   #类型0的标签
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)                    #类型1的标签

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
print(x.size())
y = torch.cat((y0, y1)).type(torch.LongTensor)
print(y.size())
# plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c = y.data.numpy(), s = 50, cmap = 'RdYlGn')
# plt.show()

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.output(x)
        return x
    
net = Net(2, 10, 2)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.02)
loss_func = nn.CrossEntropyLoss()
target_y = y.data.numpy()

for i in range(1000):
    result = net(x)
    
    loss = loss_func(result, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (i % 20 == 0):
        #plt.cla()
        prediction = torch.max(F.softmax(result, dim = 1), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c = pred_y, s = 30, cmap = 'RdYlGn')
        accuracy = sum(pred_y == target_y) / 200 # 统计预测的准确率
        # plt.text(1.5, -4, 'Accuracy = %.2f' % accuracy, fontdicct = {'size':20, 'color':'red'})
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 15, 'color':  'red'})
        plt.show()
# plt.ioff()
# plt.show()
    
    

