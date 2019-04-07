import torch
from transition_network import TransitionCNN
from utilities import MovingAverage, RunningAverage, reshapeLabels, reshapeTransitionBatch, desired_labels, save_checkpoint, get_and_write_transition_distribution
from TransitionDataSet import TransitionDataSet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import os
import pandas as pd
import numpy as np

# train model

#load the train and validation sets

train_set = TransitionDataSet('train')
valid_set = TransitionDataSet('valid')

print(str(train_set))
# print(str(valid_set))

train_loader = DataLoader(train_set, batch_size=10, num_workers=2, shuffle=True)
print('train loader length:', len(train_loader))

valid_loader = DataLoader(valid_set, batch_size=10,num_workers=2, shuffle=False)
# print('valid loader length:', len(valid_loader))

all_labels = []

print('getting data set transition distributions.............')
# create the json directory required for storage of the json info 
os.makedirs('jsons', exist_ok=True)

# get distribution of the transtitions in the training and validation set
get_and_write_transition_distribution(train_set, './jsons/shot_boundary_detector_even_distrib.json')
get_and_write_transition_distribution(valid_set, './jsons/shot_boundary_detector_even_distrib.json')


# change the transition framework to only return singular labels so need for the full ten anymore should speed up this section of the code
print('getting desired predictiion labels..............')
for batch in valid_loader:
    labels = reshapeLabels(batch['labels'])
    all_labels.extend(labels.cpu().numpy())
print('labels obtained.......')

all_labels = torch.tensor(all_labels, dtype=torch.int64)
# print('len all lables:', len(all_labels))

device = 'cuda'

torch.manual_seed(271828)
np.random.seed(271828)

#data loader --> same batch --> should just learn that one batch --> loss should go to zero

model = TransitionCNN()
model.to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total params', total_params)
total_params = sum(p.numel() for p in model.parameters())
print('all parameters',total_params )


optimizer = optim.SGD(model.parameters(), lr=0.01)

#create directories necessary for storing of checkpoints, csv data and model when ideal validation accuracy is reached
os.makedirs('checkpoints/shot_boundary_detector_even_distrib', exist_ok=True)
os.makedirs('csv_data', exist_ok=True)
os.makedirs('models', exist_ok=True)


# define the train loop for the CNN
def train(optimizer, model, first_epoch = 1, num_epochs=10):
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    validation_accuraies = []
    epoch_times = []
    
    old_accuracy = 0

    print('----------------------------------------------------------------------')

    for epoch in range(first_epoch, first_epoch + num_epochs):
        start_time_epoch = time.time()
        print('Epoch:', epoch)

        # put the model in train mode
        model.train()

        train_loss = MovingAverage()

        for index, batch in enumerate(train_loader):

            transitions = reshapeTransitionBatch(batch['transition'])
            transitions.to(device)

            # print('batch shape:', transition.shape)

            labels = reshapeLabels(batch['labels'])
            labels.to(device)

            # print('labels shape:', labels.shape)
            
            optimizer.zero_grad()

            # pass the transitions to the model for softmax prediction 
            predictions = model(transitions)

            loss = criterion(predictions, labels)

            # backward proopogate through CNN to compute gradients
            loss.backward()

            # update the weights of the model after backward propogation
            optimizer.step()

            #update the loss
            train_loss.update(loss)

            if index%1000 == 0:
                print('training loss at the end of batch', index, ':', train_loss)

        print('Training Loss after epoch ',epoch, ':',train_loss)
        train_losses.append(train_loss.value)

        #convert the model to its validation phase
        model.eval()

        valid_loss = RunningAverage()

        # store all the predictions made by the CNN
        y_pred = []

        with torch.no_grad():
            for j, batch in enumerate(valid_loader):

                # get the transitions and labels and reshape accordingly 
                valid_transitions = reshapeTransitionBatch(batch['transition'])
                valid_transitions.to(device)

                valid_labels = reshapeLabels(batch['labels'])
                valid_labels.to(device)

                valid_predictions = model(valid_transitions)

                loss = criterion(valid_predictions, valid_labels)

                valid_loss.update(loss)

                y_pred.extend(valid_predictions.argmax(dim=1).cpu().numpy())
                
                if j%500 == 0:
                    print('validation loss at the end of batch', j, ':', valid_loss)
        

        end_time_epoch = time.time()
        epcoh_time = end_time_epoch - start_time_epoch
        print('Epcoh Time Taken:', epcoh_time)
        epoch_times.append(epcoh_time)


        print('Validation Loss at end of epoch', epoch,':',valid_loss)
        valid_losses.append(valid_loss.value)

        print('----------------------------------------------------------------------')

        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        accuracy = torch.mean((y_pred == all_labels).float())
        validation_accuraies.append(float(accuracy)*100)
        
        # save model for best validation acciracy percentage
        if accuracy > old_accuracy:
            print('old accuracy:', float(old_accuracy))
            print('new accuracy:', float(accuracy))
            torch.save(model.state_dict(), 'models/shot_boundary_detector_even_distrib.pt')
            print('save model')
        old_accuracy = accuracy
        print('Validation accuracy for Epoch:',epoch,': {:.4f}%'.format(float(accuracy) * 100))


        print('----------------------------------------------------------------------')
        checkpoint_filename = 'checkpoints/shot_boundary_detector_even_distrib-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)


        #save the checkpoint of model

    # create a data frame for the train and validation losses and validation accuracies and epoch times
    df_train = pd.DataFrame(train_losses, columns=['train_losses'])
    df_valid = pd.DataFrame(valid_losses, columns=['valid_losses'])
    df_accuracies = pd.DataFrame(validation_accuraies, columns=['accuracies'])

    df_epoch = pd.DataFrame(epoch_times, columns=['epoch_time'])


    total = df_train.join(df_valid, how='outer')
    total = total.join(df_accuracies, how='outer')
    total = total.join(df_epoch, how='outer')

    total.to_csv('csv_data/expanded_data_set.csv', index=False)


start_time_train = time.time()
train(model=model, optimizer=optimizer, num_epochs=30)
end_time_train = time.time()
total_train_time = end_time_train - start_time_train

print('total train_time:', total_train_time)
