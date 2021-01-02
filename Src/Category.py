class MyMobileNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyMobileNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 153)
        )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x


model = MyMobileNet(my_pretrained_model=pretrained)

# In[ ]:


# train_category.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt


# from utils import Config
# from model import model
# from data import get_dataloader


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    print(inputs.shape)
                    outputs = model(inputs)

                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred == labels.data)

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)

            if phase == 'test':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model)

        # torch.save({'model':best_model_wts}, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        # print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))
        torch.save({'model': best_model_wts}, 'model.pth')
        print('Model saved at: {}'.format('model.pth'))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    plt.figure()
    plt.plot(np.arange(num_epochs), train_loss_list, label='Train')
    plt.plot(np.arange(num_epochs), val_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('./Mobilenet_new_loss.png', dpi=256)
    # plt.show()

    plt.figure()
    plt.plot(np.arange(num_epochs), train_acc_list, label='Train')
    plt.plot(np.arange(num_epochs), val_acc_list, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('./Mobilenet_new_acc.png', dpi=256)
    # plt.show()


if __name__ == '__main__':
    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'],
                                                        num_workers=Config['num_workers'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'], weight_decay=0.0001)
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')

    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'],
                dataset_size=dataset_size)
    text_gen()

    print('DONE')