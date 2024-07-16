###################################################################
################ Real-time Hand Gesture Recognition ###############
################ Computer Vision Course Project ###################
################ Rasel Ahmed Bhuiyan ##############################
################ PhD Student, University of Notre Dame ############
################ Email: rbhuiyan@nd.edu ###########################
################ Fall Semester --- September 2022 #################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import ConvNet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# from torchsummary import summary
# # Show model summary (in details)
# model = ConvNet(num_class=3)
# summary = summary(model, (3, 224, 224))
# print(summary)


def main(args):
    # Assigning device
    # For Apple Silicon
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # # If you are using CUDA uncomment the following line
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')

    # Pre-define value
    num_class = 3
    cropSize = 224
    batch_size = 8
    classes = ['A', 'B', 'C']
    # mean = [0.4363, 0.4328, 0.3291]
    # std = [0.2129, 0.2075, 0.2038]

    # Data pre-processor
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(cropSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(cropSize),
        transforms.ToTensor()
    ])

    # Load train, test and validation data with preprocessing
    trainset = ImageFolder(root=args.train_dir, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    validationset = ImageFolder(root=args.val_dir, transform=test_transform)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = ImageFolder(root=args.test_dir, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define model, loss function, and optimizer
    model = ConvNet(num_class=num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    if args.choice == "train":
        # Training
        loss_values=[]
        epochs = []
        num_epochs = 30
        best_acc = 0.0

        model.train()
        for epoch in range(num_epochs):
            total = 0.0
            training_loss = 0.0
            training_acc = 0.0
            for data in trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total += labels.size(0)

                loss = loss.detach().cpu().numpy()
                inputs = inputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                training_loss += loss

            # Calculate the overall accuracy on the validation set
            _, _, _, acc = model.evaluate(model, validationloader, device)
            # Store loss for each epoch
            epochs.append(epoch+1)
            loss_values.append(training_loss/total)

            # show training loss and validation accuracy per-epoch
            print(f'epoch: {epoch+1} training_loss: {training_loss/total} val_acc: {acc:.2f}')

            # Save the best model
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.weight_dir)

        # Plot training loss
        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training loss')
        plt.plot(np.array(epochs), np.array(loss_values), '-ro')
        plt.savefig('loss.png', facecolor='w')

    if args.choice == "test":
        # Test model on unknown data
        print(f'Loading best model from {args.weight_dir}')
        model.load_state_dict(torch.load(args.weight_dir))
        predicted, actual, scores, acc = model.evaluate(model, testloader, device)
        print(f'Accuracy on unknown data: {acc:.2f}%')

        # Plot confusion matrix
        cf_matrix = confusion_matrix(actual.numpy(), predicted.numpy())
        print(cf_matrix)
        # Per-class accuracy
        report = classification_report(actual.numpy(), predicted.numpy(), target_names=classes)
        print(report)
        # Calculate AUC and Display ROC curve
        fpr, tpr, threshold = roc_curve(actual.numpy(), scores.numpy(), pos_label=0)
        AUC = auc(fpr, tpr)

        # Plot ROC
        plt.figure()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.plot(fpr, tpr, label='AUC: ' + '{:.2f}'.format(AUC))
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.legend(loc='lower right')
        plt.title('ROC curve')
        plt.savefig('roc.png', facecolor='w')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test model")
    parser.add_argument("--train_dir", type=str, default="./handGestureData/train", help="Train dataset location")
    parser.add_argument("--test_dir", type=str, default="./handGestureData/test/", help="Test dataset location")
    parser.add_argument("--val_dir", type=str, default="./handGestureData/val/", help="Validation dataset location")
    parser.add_argument("--weight_dir", type=str, default="./ASL-Weights/best_model.pth", help="Save best model weight")
    parser.add_argument("--choice", type=str, default="test", help="train/test")
    args = parser.parse_args()
    main(args)