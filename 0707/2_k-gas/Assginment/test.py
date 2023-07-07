# test.py
import yaml
import torch
from torch.nn import MSELoss
from utils.model_mlp import MLP
from utils.dataset import GasDataset
from torch.utils.data import DataLoader

# Load the configuration file
with open('/NasData/home/lsh/mlcl/MLCL_2023/2_k-gas/Assginment/configs/configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
criterion = MSELoss()

# TODO: Initialize the model and load the saved model weights
model = MLP(config['model']['input_size'],config['model']['hidden_size'],config['model']['output_size'])
model.load_state_dict(torch.load(config['paths']['model_save']))

# Create the test dataset and dataloader
test_dataset = GasDataset(config['paths']['data'], start_year=2020, end_year=2020)
test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Testing
model.eval()
test_loss = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss.append(loss.item())
        # print(f'test_loss : ', {sum(test_loss)}/{len(test_loader)})
        # TODO: Implement the testing step here

# print(f'Test Loss: {test_loss.item()}')
print(f'Test Loss: {sum(test_loss) / len(test_loss)}')