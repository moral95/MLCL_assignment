# training.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler

from PIL import Image
from utils.dataset import HeadGearDataset
from models.resnet_50 import resnet50
from models.model_mlp import MLP
from utils.config import load_config
# from utils.transforms import *
from utils.plot import *
from sklearn.metrics import f1_score

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(64),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
    ])                    


def train_one_epoch(model, criterion, optimizer, dataloader, device, grad_clip):
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    all_labels = []
    all_predictions = []

    print(dataloader)
    for data, target in tqdm(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)

        _, predicted = torch.max(output.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        all_labels.extend(target.detach().cpu().numpy().tolist())
        all_predictions.extend(predicted.detach().cpu().numpy().tolist())
        loss.backward()
        
        if grad_clip is not None:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = (correct_predictions / total_predictions) * 100.0
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, criterion, dataloader, device):
    model.eval()

    running_valid_loss = 0.0
    total_valid_predictions = 0.0
    correct_valid_predictions = 0.0
    all_valid_labels = []
    all_valid_predictions = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_valid_loss += loss.item() * data.size(0)

            _, predicted = torch.max(output.data, 1)
            total_valid_predictions += target.size(0)
            correct_valid_predictions += (predicted == target).sum().item()

            all_valid_labels.extend(target.detach().cpu().numpy().tolist())
            all_valid_predictions.extend(predicted.detach().cpu().numpy().tolist())

    valid_loss = running_valid_loss / len(dataloader.dataset)
    valid_acc = (correct_valid_predictions / total_valid_predictions) * 100.0
    valid_f1 = f1_score(all_valid_labels, all_valid_predictions, average='macro')

    return valid_loss, valid_acc, valid_f1, data, output, target


def training(config):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f'We used the ({device}')

    train_data  = HeadGearDataset(config['paths']['annotation'],config['paths']['dataset_path'], mode='train', transform=transform)
    train_loader = DataLoader(train_data, batch_size = config['training']['batch_size'], shuffle = True)

    valid_data  = HeadGearDataset(config['paths']['annotation'],config['paths']['dataset_path'], mode='valid', transform=transform)
    valid_loader = DataLoader(valid_data, batch_size = config['training']['batch_size'], shuffle = False)
    # plot_dataset_distribution(config['paths']['annotation'])
    
    model = resnet50(config['model']['num_classes'])
    # model = MLP(20, 17, config['model']['num_classes'])
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    # lr_scheduler = torch.optim.lr_scheduler()
    grad_clip = config['training']['grad_clip']
   
    for epoch in range(config['training']['num_epochs']):
        # TODO: Train the model on the training data
        train_loss, train_acc, train_f1 = train_one_epoch(model, criterion, optimizer, train_loader, device, grad_clip)
        print(f"Epoch: {epoch+1}/{config['training']['num_epochs']}.. Training Loss: {train_loss:.4f}.. Training Accuracy: {train_acc:.2f}%.. Training F1 Score: {train_f1:.2f}")

        # TODO: Validate the model on the validation data
        valid_loss, valid_acc, valid_f1, data, output, target = validate(model, criterion, valid_loader, device)
        print(f"Epoch: {epoch+1}/{config['training']['num_epochs']}.. Validation Loss: {valid_loss:.4f}.. Validation Accuracy: {valid_acc:.2f}%.. Validation F1 Score: {valid_f1:.2f}")

    torch.save(model.state_dict(), config['paths']['model_save_path'])
    # plot_image(image_paths, data, output, target)
if __name__ == "__main__":
    config = load_config('configs/configs.yaml')  # specify the path to your config file
    training(config)
