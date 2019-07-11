from torchvision import datasets
import os, torch, torchvision
import torch

# redefine the MNIST dataset
class MyMNIST(datasets.MNIST):
     def __init__(self, root, digit_class, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        data, targets = torch.load(os.path.join(self.processed_folder, data_file))
        data_list = []
        target_list = []
        for i in range(len(digit_class)):
            idx = targets==digit_class[i]
            data_list.append(data[idx])
            target_list.append(torch.ones_like(targets[idx])+i-1)
        self.data = torch.cat(data_list)
        self.targets = torch.cat(target_list)

if __name__ == "__main__":
    mydata = MyMNIST(root='~/Documents/manifold_learn/ManifoldFirstClassify/data', download=True, digit_class=[1,2],
                        transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.1307,), (0.3081,))
                                ]))
    ## data visualization 
    train_loader = torch.utils.data.DataLoader(mydata, batch_size=64, shuffle=True,
                                                )
    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(type(example_targets[0]))
    import matplotlib.pyplot as plt

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()