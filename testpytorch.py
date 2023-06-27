from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# print(device)

# def imshow(inp, title=None):
#     """Display image for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images // 2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title(f'predicted: {class_names[preds[j]]}')
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
# customised densenet
class AttentionLayer(nn.Module):
    def __init__(self, in_features, attention_features):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, attention_features),
            nn.ReLU(inplace=True),
            nn.Linear(attention_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Flatten the feature maps
        x = x.view(x.size(0), -1)

        # Compute attention weights
        attention_weights = self.attention(x)

        # Apply attention weights
        x = x * attention_weights

        return x


class DenseNet201WithAttention(nn.Module):
    def __init__(self, attention_features, num_classes):
        super(DenseNet201WithAttention, self).__init__()
        self.densenet = models.densenet201(pretrained=True)

        # Replace the last layer with an attention layer
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        self.attention_layer = AttentionLayer(num_features, attention_features)

        # Add a new classifier layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.densenet(x)
        #attended_features = self.attention_layer(features)
        output = self.classifier(features)
        return output


if __name__ == '__main__':

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'C:/Users/csitaula/Desktop/onedrive/Aerial/NWPU_/2_8/'
    for i in range(0, 5):

        data_dirs = data_dir + '/' + str(i + 1) + '/'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dirs, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                                      shuffle=True, num_workers=1)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # model_ft = models.resnet50(pretrained=True)
        model_ft = DenseNet201WithAttention(attention_features=512, num_classes=len(class_names))
        print(model_ft)

        # freeze the layers
        for param in model_ft.parameters():
            param.requires_grad = True
        #
        # num_ftrs = model_ft.fc.in_features  # classifier or fc
        # # Here the size of each output sample is set to 2.
        # # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        # model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        # print(model_ft)
        criterion = nn.CrossEntropyLoss()

        model_ft = model_ft.to(device)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=50)
        print('Finished for fold:' + str(i + 1))
        print(data_dirs)



