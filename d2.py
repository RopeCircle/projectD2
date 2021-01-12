import SimpleITK as sitk
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import torch.nn.functional as F


class dataset(Dataset):
    def __init__(self, root):
        self.image_file = root + "Patient_01.nii"
        self.lable_file = root + "GT.nii"

    def __getitem__(self, item):
        data = sitk.GetArrayFromImage(sitk.ReadImage(self.image_file))[item+50, :, :]
        ra = np.max(abs(data))
        imge = data / ra
        lab = sitk.GetArrayFromImage(sitk.ReadImage(self.lable_file))[item+50, :, :]
        return imge, lab

    def __len__(self):
        return 131


path = "./data_source/Patient_01/"
train_dataset = dataset(path)
train_loader = DataLoader(train_dataset)

itk_img = sitk.ReadImage('./data_source/Patient_01/Patient_01.nii')
test = torch.from_numpy(sitk.GetArrayFromImage(itk_img))
test = torch.unsqueeze(test[130, :, :], dim=0)
test = torch.unsqueeze(test, dim=0)


class fcn(torch.nn.Module):
    def __init__(self):
        super(fcn, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 1)
        self.conv1_1 = torch.nn.Conv2d(32, 32, 1)

        self.conv2 = torch.nn.Conv2d(32, 64, 1)
        self.conv2_1 = torch.nn.Conv2d(64, 64, 1)

        self.conv3 = torch.nn.Conv2d(64, 96, 1)
        self.conv3_1 = torch.nn.Conv2d(96, 96, 1)

        self.conv4 = torch.nn.Conv2d(96, 128, 1)
        self.conv4_1 = torch.nn.Conv2d(128, 128, 1)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv1t = torch.nn.ConvTranspose2d(128, 128, 1, stride=2, output_padding=1)
        self.conv2t = torch.nn.ConvTranspose2d(128, 64, 1, stride=2, output_padding=1)
        self.conv3t = torch.nn.ConvTranspose2d(64, 32, 1, stride=2, output_padding=1)
        self.conv5 = torch.nn.Conv2d(32, 8, 1)
        self.conv6 = torch.nn.Conv2d(8, 5, 1)

    def forward(self, x):
        x_32 = F.relu(self.conv1(x))  # 32
        x_32 = F.relu(self.conv1_1(x_32))  # 32
        x = self.pool(x_32)
        x_64 = F.relu(self.conv2(x))  # 64
        x_64 = F.relu(self.conv2_1(x_64))  # 64
        x = self.pool(x_64)
        x_96 = F.relu(self.conv3(x))  # 96
        x_96 = F.relu(self.conv3_1(x_96))  # 96
        x = self.pool(x_96)
        x_128 = F.relu(self.conv4(x))  # 128
        x_128 = F.relu(self.conv4_1(x_128))  # 128
        x_t_128 = F.relu(self.conv1t(x_128))# + x_96
        x_t_64 = F.relu(self.conv2t(x_t_128)) + x_64
        x_t_32 = F.relu(self.conv3t(x_t_64)) + x_32
        x_8 = F.relu(self.conv5(x_t_32))
        x_5 = F.relu(self.conv6(x_8))
        return x_5


def train():
    for epoch in range(80):
        NET = net.train()
        train_loss = 0
        for data in train_loader:
            img = torch.autograd.Variable(torch.unsqueeze(data[0], dim=1).float().cuda())
            label = torch.autograd.Variable(data[1].cuda()).long()

            out = NET(img)
            loss = loss_function(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data

        scheduler.step()
        print('Epoch:' + str(epoch) + ',Train Loss:' + str(train_loss/len(train_loader)))
        rt = NET(test.float().cuda())
        rt = rt.max(dim=1)[1].data.cpu().numpy()
        print(np.max(rt), np.min(rt))


net = fcn().cuda()
loss_function = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1000, 1000, 1000, 1000]).cuda())
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, last_epoch=-1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
train()
torch.save(net, 'net.pkl')


net = torch.load('net.pkl')
r = net(test.float().cuda())
r = r.max(dim=1)[1].data.cpu().numpy()
print(np.max(r))
plt.imshow(r[0, :, :])
plt.show()





