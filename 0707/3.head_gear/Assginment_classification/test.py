# test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import HeadGearDataset
from models.resnet_50 import resnet50
from utils.config import load_config
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

def test(config):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # TODO: Load test dataset using the HeadGearDataset class
    # TODO: Define data loader for test set
    test_data  = HeadGearDataset(config['paths']['annotation'],config['paths']['dataset_path'], mode='test', target_transform=transform)
    test_loader = DataLoader(test_data, batch_size = config['training']['batch_size'], shuffle = False)

    # TODO: Define the model and load the trained weights
    model = resnet50(config['model']['num_classes'])
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    # # fill this in
    model.to(device)
    
    # TODO: Set the model to test mode
    # model.'# fill this in'
    model.eval()

    total_predictions = 0.0
    correct_predictions = 0.0
    all_labels = []
    all_predictions = []

    # TODO: Stop tracking gradients
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # TODO: Define Output
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_predictions.extend(predicted.detach().cpu().numpy().tolist())

    test_acc = (correct_predictions / total_predictions) * 100.0
    test_f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Test Accuracy: {test_acc:.2f}%.. Test F1 Score: {test_f1:.2f}")

if __name__ == "__main__":
    config = load_config('./configs/configs.yaml')
    test(config)
