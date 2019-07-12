import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def visualize_logits(input, target, logits, digits):
    """
    The input should be tuple of numpy array and the target should be numpy array.
    logits should be ((logits1, logits2,...), (perturb1, perturb2,...))
    """
    if len(digits) == 2:
        DIM = 2
    elif len(digits) == 3:
        DIM = 3
    else:
        raise AssertionError('Can only visualize at most 3D.')
    # print(len(logits), logits[0].shape)
    if DIM==2:
        _, axes = plt.subplots(len(logits), logits[0].shape[1], squeeze=False)
    else:
        _, axes = plt.subplots(len(logits), logits[0].shape[1], subplot_kw=dict(projection='3d'), squeeze=False)

    for i in range(len(logits)):
        for j in range(logits[0].shape[1]):
            for index, k in enumerate(digits):
                label = '%d' % k if i==0 else '%d attack' % k
                axes[i,j].scatter(*(logits[i][:, j, :][target==index][:,m] for m in range(DIM)), 
                                label=label, alpha=0.3)
            # axes[i,j].plot(x_dec, x_dec)
            axes[i,j].legend()
            axes[i,j].axis('equal')
    plt.show()

def visualize_imgs(input, perturb, pred, target):
    att_s_index = pred != target
    att_f_index = pred == target
    # print(pred_list_att.shape, target_list_test.shape)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(4,6,i+1)
        plt.tight_layout()
        plt.imshow(input[att_s_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+7)
        plt.tight_layout()
        plt.imshow(perturb[att_s_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+13)
        plt.tight_layout()
        plt.imshow(input[att_f_index][i,0,], cmap='gray', interpolation='none')
        plt.subplot(4,6,i+19)
        plt.tight_layout()
        plt.imshow(perturb[att_f_index][i,0,], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    
    plt.show()