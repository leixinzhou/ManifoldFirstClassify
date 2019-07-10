import torch
import torchvision
from nets import AlexNet
from dataset import MyMNIST
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

n_epochs = 20
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
digit = [0,1]
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  MyMNIST('~/Documents/manifold_learn/ManifoldFirstClassify/data', train=True, download=True, digit_class=digit,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  MyMNIST('~/Documents/manifold_learn/ManifoldFirstClassify/data', train=False, download=True, digit_class=digit,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

network = AlexNet()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output, _ = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model.pth')
    #   torch.save(optimizer.state_dict(), '~/Documents/manifoldlearn/ManifoldFirstClassify/results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output, _ = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def logits_vis(epoch):
    network.load_state_dict(torch.load('results/model.pth'))
    network.eval()
    logits_list_test = []
    target_list_test = []
    logits_list_att = []
    correct = 0
    for data, target in test_loader:
        _, logits = network(data)
        logits_list_test.append(logits.detach().numpy())
        target_list_test.append(target.detach().numpy())
        data.requires_grad = True
        output, _ = network(data)
        loss = F.nll_loss(output, target)
        network.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, 1., data_grad)
        _, logits = network(perturbed_data)
        logits_list_att.append(logits.detach().numpy())
        pred = logits.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    print('Accuracy: {:.0f}%\n'.format(
        100. * correct / len(test_loader.dataset)))
    logits_list_test = np.concatenate(logits_list_test)
    logits_list_att = np.concatenate(logits_list_att)
    target_list_test = np.concatenate(target_list_test)
    zero_list_test = logits_list_test[target_list_test==0]
    one_list_test = logits_list_test[target_list_test==1]
    zero_list_att = logits_list_att[target_list_test==0]
    one_list_att = logits_list_att[target_list_test==1]
    
    _, ax = plt.subplots(1,2, sharex=True, sharey=True)
    ax[0].scatter(one_list_test[:,0], one_list_test[:,1],  label='1 test', alpha=0.3)
    ax[0].scatter(zero_list_test[:,0], zero_list_test[:,1],  label='0 test', alpha=0.3)
    ax[1].scatter(one_list_att[:,0], one_list_att[:,1],  label='1 att', alpha=0.3)
    ax[1].scatter(zero_list_att[:,0], zero_list_att[:,1],   label='0 att', alpha=0.3)
    ax[0].legend()
    ax[1].legend()
    plt.show()

# test()
# for epoch in range(1, n_epochs + 1):
#   train(epoch)
#   test()
# fig = plt.figure()

# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# plt.show()
logits_vis(1)
