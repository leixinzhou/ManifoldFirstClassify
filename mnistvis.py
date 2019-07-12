import torch
import torchvision
from nets import AlexNet
from dataset import MyMNIST
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from loss import DoubleCELoss
import math
from vis import *
from attack import *
import yaml, argparse
import os


def train(network, loss_f, optimizer, input, target):
  network.train()
  optimizer.zero_grad()
  output = network(input)
  pred = output[:, 0,].data.max(1, keepdim=True)[1]
  correct = pred.eq(target.data.view_as(pred)).float().mean()
  loss = loss_f(output, target)
  loss.backward()
  optimizer.step()
  return loss.detach().numpy(), correct.numpy()

def val(network, loss_f, input, target):
  network.eval()
  output = network(input)
  pred = output[:, 0,].data.max(1, keepdim=True)[1]
  correct = pred.eq(target.data.view_as(pred)).float().mean()
  loss = loss_f(output, target)
  return loss.detach().numpy(), correct.numpy()

def learn(network, loss_f, hps):
  train_loader = torch.utils.data.DataLoader(
    MyMNIST('~/Documents/manifold_learn/ManifoldFirstClassify/data', train=True, download=True, digit_class=hps['digit'],
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=hps['batch_size_train'], shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    MyMNIST('~/Documents/manifold_learn/ManifoldFirstClassify/data', train=False, download=True, digit_class=hps['digit'],
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=hps['batch_size_test'], shuffle=True)
  optimizer = optim.SGD(network.parameters(), lr=hps['learning_rate'],
                      momentum=hps['momentum'])
  best_loss = hps['best_loss']
  tr_loss_list = []
  tr_acc_list = []
  val_loss_list = []
  val_acc_list = []
  for epoch in range(1, hps['n_epochs']+1):
    train_loss = 0
    train_acc = 0
    train_mb = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
      loss, acc = train(network, loss_f, optimizer, data, target)
      train_loss += loss
      train_acc += acc
      train_mb += 1
    epoch_tr_loss = train_loss / train_mb
    epoch_tr_acc = train_acc / train_mb
    tr_loss_list.append(epoch_tr_loss)
    tr_acc_list.append(epoch_tr_acc)
    print("Epoch: " + str(epoch) + " tr_loss: " + "%.3e" % epoch_tr_loss  + " tr_acc: " + "%.3e" % epoch_tr_acc)
    val_loss = 0
    val_acc = 0
    val_mb = 0
    for batch_idx, (data, target) in enumerate(test_loader):
      loss, acc = val(network, loss_f, data, target)
      val_loss += loss
      val_acc += acc
      val_mb += 1
    epoch_val_loss = val_loss / val_mb
    epoch_val_acc = val_acc / val_mb
    val_loss_list.append(epoch_val_loss)
    val_acc_list.append(epoch_val_acc)
    print("Epoch: " + str(epoch) + " val_loss: " + "%.3e" % epoch_val_loss + " val_acc: " + "%.3e" % epoch_val_acc)
    if epoch_val_loss < best_loss:
      best_loss = epoch_val_loss
      torch.save(network.state_dict(), os.path.join(hps['save_dir'], 'model_%s.pth' % hps['degree']))
      #   torch.save(optimizer.state_dict(), '~/Documents/manifoldlearn/ManifoldFirstClassify/results/optimizer.pth')

  fig = plt.figure()
  plt.plot(tr_loss_list, label='tr loss')
  plt.plot(tr_acc_list, label='tr acc')
  plt.plot(val_loss_list, label='val loss')
  plt.plot(val_acc_list, label='val acc')
  plt.legend()
  plt.show()
  plt.savefig(os.path.join(hps['save_dir'], 'tr_val_degree_%.2f.png' %  hps['degree']), bbox_inches='tight')
  plt.close()


def logits_vis(network, loss_f, hps):
  test_loader = torch.utils.data.DataLoader(
    MyMNIST('~/Documents/manifold_learn/ManifoldFirstClassify/data', train=False, download=True, digit_class=hps['digit'],
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=hps['batch_size_test'], shuffle=True)

  network.load_state_dict(torch.load(os.path.join(hps['save_dir'], 'model_%s.pth' % hps['degree'])))
  network.eval()
  logits_list_test = []
  target_list_test = []
  logits_list_att = []
  data_list_test = []
  perturbed_data_list_test = []
  pred_list_att = []
  correct = 0
  for data, target in test_loader:
      logits = network(data)
      logits_list_test.append(logits.detach().numpy())
      target_list_test.append(target.detach().numpy())
      data_list_test.append(data.detach().numpy())
      data.requires_grad = True
      # output, _ = network(data)
      # loss = F.nll_loss(output, target)
      output = network(data)
      loss = loss_f(output, target)
      network.zero_grad()
      loss.backward()
      data_grad = data.grad.data
      perturbed_data = fgsm_attack(data, hps['epsilon'], data_grad)
      perturbed_data_list_test.append(perturbed_data.detach().numpy())
      logits = network(perturbed_data)
      logits_list_att.append(logits.detach().numpy())
      pred = logits[:, 0, :].data.max(1, keepdim=True)[1]
      pred_list_att.append(pred.detach().squeeze().numpy())
      correct += pred.eq(target.data.view_as(pred)).sum().float()
  acc = correct / len(test_loader.dataset)
  print('Accuracy: %.3e' % acc)
  # print('Accuracy: {:.0f}%\n'.format(
    #     100. * correct_2 / len(test_loader.dataset)))
  logits_list_test = np.concatenate(logits_list_test, axis=0)
  logits_list_att = np.concatenate(logits_list_att, axis=0)
  target_list_test = np.concatenate(target_list_test, axis=0)
  pred_list_att = np.concatenate(pred_list_att, axis=0)
  perturbed_data_list_test = np.concatenate(perturbed_data_list_test, axis=0)
  data_list_test = np.concatenate(data_list_test, axis=0)
  visualize_logits(data_list_test, target_list_test, (logits_list_test, logits_list_att), digits=hps['digit'],\
                      hps=hps)
  visualize_imgs(data_list_test, perturbed_data_list_test, pred_list_att, target_list_test,\
                      hps=hps)
    

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-hp', '--hyperparams', default='./para/hparas_unet.json',
                      type=str, metavar='FILE.PATH',
                      help='path to hyperparameters setting file (default: ./para/hparas_unet.json)')
  args = parser.parse_args()
  try:
      with open(args.hyperparams, "r") as config_file:
          hps = yaml.load(config_file)
  except IOError:
      print('Couldn\'t read hyperparameter setting file')
  
  torch.backends.cudnn.enabled = False
  torch.manual_seed(hps['random_seed'])
  degree = math.pi*hps['degree']
  network = AlexNet(nb_class=len(hps['digit']), degree=degree)
  loss_f = DoubleCELoss()
  
  if hps['train_mode']:
    learn(network, loss_f,  hps)
  else:
      logits_vis(network, loss_f,  hps)

if __name__ == '__main__':
    main()