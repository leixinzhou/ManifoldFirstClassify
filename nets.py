import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = torch.norm(x, dim=1)
        # return F.log_softmax(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, nb_class, degree):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nb_class)
        self.degree = degree

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        rot_x = BatchRotLogits(x, self.degree)
        # x = torch.norm(x, dim=1)
        # return F.log_softmax(x), x
        return x, rot_x
def BatchRotLogits(input, degree):
    center = torch.mean(input, dim=0)
    input -= center
    rot_mat = torch.tensor([[math.cos(degree), math.sin(degree)], [-math.sin(degree), math.cos(degree)]])
    input = torch.matmul(input, rot_mat)
    return input + center

# a = torch.tensor([[-1, 1.], [-2, 2], [-3, 3]])
# print(BatchRotLogits(a))