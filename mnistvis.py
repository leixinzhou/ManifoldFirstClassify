import torch
import torchvision
from nets import AlexNet
from dataset import MyMNIST
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from loss import DoubleCELoss
import math

train_mode = True
n_epochs = 50
batch_size_train = 512
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
digit = [1,7]
epsilon = 1.
random_seed = 1
degree = 0.5*math.pi
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

network = AlexNet(nb_class=len(digit), degree=degree)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

loss_f = DoubleCELoss()
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = loss_f(*output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model_%s_dce_cent.pth' % ''.join(str(e) for e in digit))
    #   torch.save(optimizer.state_dict(), '~/Documents/manifoldlearn/ManifoldFirstClassify/results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += loss_f(*output, target, size_average=False).item()
      pred = output[0].data.max(1, keepdim=True)[1]
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
    network.load_state_dict(torch.load('results/model_%s_dce_cent.pth' % ''.join(str(e) for e in digit)))
    network.eval()
    logits_list_test_1 = []
    logits_list_test_2 = []
    target_list_test = []
    logits_list_att_1 = []
    logits_list_att_2 = []
    data_list_test = []
    perturbed_data_list_test = []
    pred_list_att_1 = []
    pred_list_att_2 = []
    correct_1 = 0
    correct_2 = 0
    for data, target in test_loader:
        logits1, logits2 = network(data)
        logits_list_test_1.append(logits1.detach().numpy())
        logits_list_test_2.append(logits2.detach().numpy())
        target_list_test.append(target.detach().numpy())
        data_list_test.append(data.detach().numpy())
        data.requires_grad = True
        # output, _ = network(data)
        # loss = F.nll_loss(output, target)
        output = network(data)
        loss = loss_f(*output, target)
        network.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        perturbed_data_list_test.append(perturbed_data.detach().numpy())
        logits1, logits2 = network(perturbed_data)
        logits_list_att_1.append(logits1.detach().numpy())
        logits_list_att_2.append(logits2.detach().numpy())
        pred_1 = logits1.data.max(1, keepdim=True)[1]
        pred_list_att_1.append(pred_1.detach().squeeze().numpy())
        pred_2 = logits2.data.max(1, keepdim=True)[1]
        pred_list_att_2.append(pred_2.detach().squeeze().numpy())
        correct_1 += pred_1.eq(target.data.view_as(pred_1)).sum()
        correct_2 += pred_2.eq(target.data.view_as(pred_2)).sum()
    print('Accuracy: {:.0f}%\n'.format(
        100. * correct_1 / len(test_loader.dataset)))
    print('Accuracy: {:.0f}%\n'.format(
        100. * correct_2 / len(test_loader.dataset)))
    logits_list_test_1 = np.concatenate(logits_list_test_1)
    logits_list_att_1 = np.concatenate(logits_list_att_1)
    logits_list_test_2 = np.concatenate(logits_list_test_2)
    logits_list_att_2 = np.concatenate(logits_list_att_2)
    target_list_test = np.concatenate(target_list_test)
    pred_list_att = np.concatenate(pred_list_att_1)
    perturbed_data_list_test = np.concatenate(perturbed_data_list_test)
    data_list_test = np.concatenate(data_list_test)
    # decision boundary
    x_dec = np.linspace(-5,5, 50)

    # ax = fig.add_subplot(121, projection='3d')
    _, axes = plt.subplots(2,2)
    for i in range(len(digit)):
        axes[0,0].scatter(*(logits_list_test_1[target_list_test==i][:,j] for j in range(len(digit))), 
                        label='%d test 1' % digit[i], alpha=0.3)
        axes[0,0].plot(x_dec, x_dec)
    axes[0,0].legend()
    axes[0,0].axis('equal')
    for i in range(len(digit)):
        axes[0,1].scatter(*(logits_list_test_2[target_list_test==i][:,j] for j in range(len(digit))), 
                        label='%d test 2' % digit[i], alpha=0.3)
        axes[0,1].plot(x_dec, x_dec)
    axes[0,1].legend()
    axes[0,1].axis('equal')
    # ax = fig.add_subplot(122, projection='3d')

    for i in range(len(digit)):
        axes[1,0].scatter(*(logits_list_att_1[target_list_test==i][:,j] for j in range(len(digit))), 
                        label='%d test att 1' % digit[i], alpha=0.3)
        axes[1,0].plot(x_dec, x_dec)
    axes[1,0].legend()
    axes[1,0].axis('equal')
    for i in range(len(digit)):
        axes[1,1].scatter(*(logits_list_att_2[target_list_test==i][:,j] for j in range(len(digit))), 
                        label='%d test att 2' % digit[i], alpha=0.3)
        axes[1,1].plot(x_dec, x_dec)
    axes[1,1].legend()
    axes[1,1].axis('equal')
    plt.show()
    
#     _, ax = plt.subplots(1,2, sharex=True, sharey=True)
#     ax[0].scatter(one_list_test[:,0], one_list_test[:,1],  label='d% test' % digit[], alpha=0.3)
#     ax[0].scatter(zero_list_test[:,0], zero_list_test[:,1],  label='0 test', alpha=0.3)
#     ax[1].scatter(one_list_att[:,0], one_list_att[:,1],  label='1 att', alpha=0.3)
#     ax[1].scatter(zero_list_att[:,0], zero_list_att[:,1],   label='0 att', alpha=0.3)
#     ax[0].legend()
#     ax[1].legend()
#     plt.show()
    att_s_index = pred_list_att!=target_list_test
    att_f_index = pred_list_att==target_list_test
    # print(pred_list_att.shape, target_list_test.shape)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(4,6,i+1)
        plt.tight_layout()
        plt.imshow(data_list_test[att_s_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+7)
        plt.tight_layout()
        plt.imshow(perturbed_data_list_test[att_s_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+13)
        plt.tight_layout()
        plt.imshow(data_list_test[att_f_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+19)
        plt.tight_layout()
        plt.imshow(perturbed_data_list_test[att_f_index][i,0,], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    
    plt.show()
if train_mode:
    test()
    for epoch in range(1, n_epochs + 1):
      train(epoch)
      test()
    fig = plt.figure()

    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
else:
    logits_vis(1)
