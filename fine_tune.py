import torch
import torch.nn as nn
import numpy as np
import os


def train_fine_tune(net,train_data_loader,val_data_loader,epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr=.0001)
    train_tune_model(net, optimizer, criterion, train_data_loader, val_data_loader, epoch, 'net_tuned')


def train_one_epoch(model, optimizer, criterion, train_data_loader, val_data_loader):
    train_loss = 0
    val_loss = 0

    model.train()
    for images, labels in train_data_loader:

        images = images.cuda()
        labels = labels.cuda()
        labels = labels.long()
        optimizer.zero_grad()
        out,feature = model(images)
        # print('-'*10)
        # print(labels.dtype)
        # print(labels.type())
        # print('-'*10)

        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    for images, labels in val_data_loader:

        images = images.cuda()
        labels = labels.cuda()
        labels = labels.long()
        out,feature = model(images)
        loss = criterion(out, labels)

        val_loss += loss.item()
    train_loss = train_loss / len(train_data_loader.dataset)
    val_loss = val_loss / len(val_data_loader.dataset)
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(train_loss, val_loss))

    return train_loss, val_loss


def train_tune_model(model, optimizer, criterion, train_data_loader, val_data_loader, epochs, model_name):
    no_improvement = 0
    best_loss = np.inf
    train_losses = []
    val_losses = []

    for i, epoch in enumerate(range(1, epochs + 1)):
        train_loss, val_loss = train_one_epoch(model, optimizer, criterion, train_data_loader, val_data_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        p_count = 0
        for p in model.feature.named_parameters():
            if p_count >= 16 - i * 2:
                p[1].requires_grad = True
                # print('Tracking Gradient For:', p[0])
            p_count += 1

        # Saving the weights of the best model according to validation score
        if val_loss < best_loss:
            no_improvement = 0
            best_loss = val_loss
            if not os.path.exists('fine_tune_models'):
                os.mkdir('fine_tune_models')
            # print('Improved Model Score - Updating Best Model Parameters...')
            torch.save(model.state_dict(), f'./fine_tune_models/{model_name}.pt')
        else:
            no_improvement += 1
            if no_improvement == 5:
                # print('No Improvement for 5 epochs, Early Stopping')
                break
