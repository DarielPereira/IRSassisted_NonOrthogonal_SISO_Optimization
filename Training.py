"""
This script trains a neural network model using either supervised or self-supervised learning.
"""

import os
import torch as th
from torch.utils.data import DataLoader, random_split
from functionsNN import supervised_Dataset, self_supervised_Dataset, NN_supervised

train_mode = 'supervised' # 'supervised' or 'self_supervised'
load_model = False
model = 'NN1'
learning_rate = 0.0001
num_epochs = 5
trainingConfiguration = f'_{train_mode}_Epochs_{num_epochs}_{model}_{learning_rate}.pt'

dataConfiguration = 'M_32_K_5_setups_200_realiz_1000'
dataset_directory = f'./TrainingData/Dataset_{dataConfiguration}'
model_directory = f'./TrainingData/Model_{dataConfiguration}'

# Load the dataset
match train_mode:
    case 'supervised':
        dataset = supervised_Dataset()
        try:
            dataset.load(dataset_directory + '_supervised.pt')
        except FileNotFoundError:
            print('No stored training data found')
    case 'self_supervised':
        dataset = self_supervised_Dataset()
        try:
            dataset.load(dataset_directory + '_self_supervised.pt')
        except FileNotFoundError:
            print('No stored training data found')
    case _:
        raise ValueError('Invalid training mode')

# Set a seed for reproducibility
th.manual_seed(0)

# Model, optimizer, and loss
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
input_size = 2*dataset[0][0].numel()  # Flattened input size
output_size = 2*dataset[0][1].numel()  # Flattened output size
model = NN_supervised(input_size, output_size).to(device)

if load_model:
    model.load_model(f'./TrainingData/'+ 'Model_M_32_K_5_setups_200_realiz_1000_supervised_Epochs_5_NN1_0.0001.pt')

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = th.nn.MSELoss()

# Split dataset into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Dataloaders
match train_mode:
    case 'supervised':
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Validate before training
        val_loss = model.model_validate(val_loader, loss_fn)
        print(f"Before training validation Loss: {val_loss:.4f}")

        # Train the model
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            train_loss = model.model_train(train_loader, optimizer, loss_fn, train_mode)
            val_loss = model.model_validate(val_loader, loss_fn)
            print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    case 'self_supervised':
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Train the model
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')

            train_loss = model.model_train(train_loader, optimizer, loss_fn, train_mode, dataset.dict_conf)
            print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}")

    case _:
        raise ValueError('Invalid training mode')



model.save_model(model_directory+trainingConfiguration)