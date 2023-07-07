# training.py
import yaml
import torch
import torch.optim as optim
from torch.nn import MSELoss
from utils.model_mlp import MLP
from utils.dataset import GasDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open('/NasData/home/lsh/mlcl/MLCL_2023/2_k-gas/Assginment/configs/configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# TODO: Initialize the model, loss, and optimizer
model = MLP(config['model']['input_size'],config['model']['hidden_size'],config['model']['output_size'])
print(model)
criterion = MSELoss()
optimizer = optim.Adam(params=model.parameters(),lr = config['training']['learning_rate'])

# Create the training and validation datasets and dataloadersft 
train_dataset = GasDataset(config['paths']['data'], start_year=None, end_year=2017)
train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

val_dataset = GasDataset(config['paths']['data'], start_year=2018, end_year=2019)
val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Training
for epoch in range(config['training']['num_epochs']):
    # Training phase
    model.train()
    train_losses = []
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        train_losses.append(loss.item())
        # train_losses += loss.item()
        # n_correct += (torch.max(outputs.data, 1) == labels).sum().items()
        # n_train += targets.size(0) 
        # print(f'train_loss : ' , {train_losses}/{train_loader})

        # TODO: Implement the training step here

    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())

            # print(f'val_loss : ', {val_losses}/{val_loader})
            # TODO: Implement the validation step here
    if (epoch+1) % 50 == 0:
        print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}, '
              f'Train Loss: {sum(train_losses)/len(train_losses)}, '
              f'Val Loss: {sum(val_losses)/len(val_losses)}')

# TODO: Save the model
torch.save(model.state_dict(), config['paths']['model_save'])