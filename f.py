import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.autograd import Variable
import SimpleITK as sitk

voc_root = 'C:/Users/JHY/Downloads/data_source'

itk_GT = sitk.ReadImage('C:/Users/JHY/Downloads/data_source/Patient_40/GT.nii')
itk_img = sitk.ReadImage('C:/Users/JHY/Downloads/data_source/Patient_40/Patient_40.nii')
img = sitk.GetArrayFromImage(itk_img)
gt = sitk.GetArrayFromImage(itk_GT)


def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


def show_image(n):
    data, label = read_images()
    im = Image.open(data[n])
    t = tfs.ToTensor()
    print(t(im).size())
    plt.imshow(im)
    plt.show()


def crop(data, label, height, width):
    box = (0, 0, width, height)
    data = data.crop(box)
    label = label.crop(box)
    return data, label


classes = ['eso', 'heart', 'trachea', 'aorta']
colormap = [[1], [2], [3], [4]]
cm2lbl = np.zeros(256 ** 3)
for i, cm in enumerate(colormap):
    cm2lbl[cm[0]*256*256 + cm[1]*256 + cm[2]] = i


def image2label(im):
    data = np.array(im, dtype='int32')
    idx = data[:, :, 0]*256*256 + data[:, :, 1]*256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64')


def image_transforms(data, label, height, width):
    data, label = crop(data, label, height, width)
    im_tfs = tfs.Compose(transforms=[tfs.ToTensor(), tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data = im_tfs(data)
    label = image2label(label)
    label = torch.from_numpy(label)
    return data, label


class VOCSegDataset(Dataset):
    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('read' + str(len(self.data_list)) + 'images')

    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label= self.transforms(img, label, self.crop_size[0], self.crop_size[1])
        return img, label

    def __len__(self):
        return len(self.data_list)
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, image_transforms)
voc_test = VOCSegDataset(False, input_shape, image_transforms)
train_data = DataLoader(voc_train, 64, shuffle=True, num_workers=4)
valid_data = DataLoader(voc_test, 128, num_workers=4)
'''
if __name__ == '__main__':
    image, label = next(iter(train_data))
    print(image[0].size())
    im = image[0].permute(1, 2, 0)
    plt.imshow(im)
    plt.show()
'''
pretrained_net = torchvision.models.resnet34()
num_classes = len(classes)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 1
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1-abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        self.stage2 = list(pretrained_net.children())[-4]
        self.stage3 = list(pretrained_net.children())[-3]

        self.scores1 = nn.Conv2d(512, num_classes, 1)
        self.scores2 = nn.Conv2d(256, num_classes, 1)
        self.scores3 = nn.Conv2d(128, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)

        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)

    def forward(self, x):
        s1 = self.stage1(x)

        s2 = self.stage2(s1)

        s3 = self.stage3(s2)

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        x1 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(x1)
        x2 = s1 + s2

        s = self.upsample_8x(x2)
        return s


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


net = fcn(num_classes)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)

for e in range(80):
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0

    net = net.train()
    if __name__ == '__main__':
        for data in train_data:
            im = Variable(data[0])
            label = Variable(data[1])

            out = torch.nn.functional.log_softmax(net(im), dim=1)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc





