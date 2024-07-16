###################################################################
################ Real-time Hand Gesture Recognition ###############
################ Computer Vision Course Project ###################
################ Rasel Ahmed Bhuiyan ##############################
################ PhD Student, University of Notre Dame ############
################ Email: rbhuiyan@nd.edu ###########################
################ Fall Semester --- September 2022 #################

import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_class):
        super(ConvNet, self).__init__()
        self.convLayers = nn.Sequential(

            # First Block of Convolution Layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Second Block of Convolution Layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding="same"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Third Block of Convolution Layer
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # Fourth Block of Convolution Layer
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(

            # First Block of Fully Connected Layer
            nn.Linear(in_features=512*14*14, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Second Block of Fully Connected Layer
            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Third Block of Fully Connected Layer
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Classification Layer
            nn.Linear(in_features=512, out_features=num_class),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.convLayers(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    # Define evaluation function to calculate model accuracy
    def evaluate(self, model, dataloader, device):
        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                # run the model on the test set to predict labels
                outputs = model(inputs)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                inputs = inputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                predicted = predicted.detach().cpu().numpy()

                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)

        return predicted, labels, accuracy
    
