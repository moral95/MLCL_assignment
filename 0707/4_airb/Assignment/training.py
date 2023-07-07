# train.py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer
from utils.dataset import ReviewDataset
from utils.config import load_config
# TODO: import precision_score, recall_score, f1_score, 
#   accuracy_score, confusion_matrix from sklearn.metrics
# TODO: import train_test_split from sklearn
from torch import binary_cross_entropy_with_logits as BCEWithLogits
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# Load the configuration
config = load_config('configs/configs.yaml')

# TODO: set the seed using the seed_everything function 
#   with the seed from the config
seed = seed_everything(config['training']['seed'])

# TODO: load the data using pandas from the path in the config
data = pd.read_csv(config['paths']['data'])

# TODO: split the data into training and validation sets
train_data, temp_data = train_test_split(data, test_size = 0.3, random_state = 0.3)     # test_size: 0.3, random_state: seed
val_data, _ = train_test_split(temp_data, test_size = 0.5, random_state = seed)               # test_size: 0.5, random_state: seed
# WARNING: the test set is not used in this 'training.py' 
# csv 파일이 사이즈가 크면, 3개의 data로 구분지어서 직접 작성해서 작성한다.

# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")    # Check HuggingFace's documentation
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")  # Check HuggingFace's documentation
print(f'model : {model}')
print(f'model.classifier : {model.classifier}')


# Get the number of features of the last layer
num_features = model.classifier.in_features

# Add a new layer without modifying the last layer
model.classifier = nn.Sequential(
    nn.Linear(num_features, num_features//2),
    nn.ReLU(),
    nn.Linear(num_features//2, 1)
)
# binary 문제에 대해서는 output을 1로 주어서 sigmoid를 사용하는 거 같다.

# TODO: initialize the dataset and dataloader
train_dataset = ReviewDataset(dataframe=data, tokenizer=tokenizer, max_len= 128)
train_loader = DataLoader(train_dataset, batch_size= config['training']['batch_size'], shuffle = True)

val_dataset = ReviewDataset(data, tokenizer, 128)
val_loader = DataLoader(val_dataset, config['training']['batch_size'], shuffle = False)

# Training settings
device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')

# TODO: set the optimizer using AdamW with the learning rate from the config
optimizer = torch.Adamw(model.parameters())
lr = config['training']['learning_rate']

# TODO: set the number of epochs from the config
num_epochs = config['training']['num_epochs']

# Initialize TensorBoard writer
writer = SummaryWriter()

# TODO: Move the model to the correct device
model = model.to(device)

# TODO: set the loss function to BCEWithLogitsLoss
criterion = BCEWithLogits()

best_val_loss = float('inf')
for epoch in range(num_epochs):
    # TODO: train the model
    model.train()
    
    epoch_loss = 0
    all_predictions = []
    all_labels = []
    for batch in train_loader:
        # TODO: zero the gradients
        optimizer.zero_grad()
        
        # TODO: move the batch each element
        input_ids = batch.ids.to(device)
        attention_mask = batch.mask.to(device)
        labels = batch.labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        # TODO: calculate the loss 'outputs.logits' and 'labels'
        # ! Check the documentation of BCEWithLogitsLoss
        # ! Check the outputs.logits and  shapes !
        print(f'outputs.logits: {outputs.logits.shape}')
        print(f'labels : {labels.shape}')
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        # TODO: calculate the predictions 'outputs.logits' 
        #   using torch.round and torch.sigmoid
        predictions = torch.round(outputs.logits)
        all_predictions.append(predictions.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    train_precision = precision_score(all_labels, all_predictions)
    train_recall = recall_score(all_labels, all_predictions)
    train_f1 = f1_score(all_labels, all_predictions)
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_confusion_matrix = confusion_matrix(all_labels, all_predictions)
    
    print('Train Confusion Matrix:\n', train_confusion_matrix)
    print(f'''
          Epoch: {epoch+1}, 
          Loss: {epoch_loss/len(train_loader):.4f}, 
          Precision: {train_precision:.4f}, 
          Recall: {train_recall:.4f}, 
          F1 Score: {train_f1:.4f}, 
          Accuracy: {train_accuracy:.4f}
        ''')
    
    writer.add_scalar('Train/Loss', epoch_loss/len(train_loader), epoch)
    writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
    writer.add_scalar('Train/Precision', train_precision, epoch)
    writer.add_scalar('Train/Recall', train_recall, epoch)
    writer.add_scalar('Train/F1 Score', train_f1, epoch)

    # Validate the model
    # TODO: evaluate the model
    model.eval()
    
    val_loss = 0
    all_predictions = []
    all_labels = []
    # TODO: no gradients
    with torch.no_grad():
    
        for batch in val_loader:
            # TODO: move the batch each element
            # input_ids = '# FILL_THIS_IN'.to(device)
            # attention_mask = '# FILL_THIS_IN'.to(device)
            # labels = '# FILL_THIS_IN'.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # TODO: calculate the loss 'outputs.logits' and 'labels'
            # ! Check the documentation of BCEWithLogitsLoss
            # ! Check the outputs.logits and labels shapes !
            # loss = criterion(# FILL_THIS_IN)

            val_loss += loss.item()
            
            # TODO: calculate the predictions 'outputs.logits' 
            #   using torch.round and torch.sigmoid
            # predictions = # FILL_THIS_IN
            all_predictions.append(predictions.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    val_precision = precision_score(all_labels, all_predictions)
    val_recall = recall_score(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_confusion_matrix = confusion_matrix(all_labels, all_predictions)
    
    print('Validation Confusion Matrix:\n', val_confusion_matrix)
    print(f'''
          Validation Loss: {val_loss/len(val_loader):.4f}, 
          Precision: {val_precision:.4f}, 
          Recall: {val_recall:.4f}, 
          F1 Score: {val_f1:.4f}, 
          Accuracy: {val_accuracy:.4f}
        ''')
    
    writer.add_scalar('Validation/Loss', val_loss/len(val_loader), epoch)
    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
    writer.add_scalar('Validation/Precision', val_precision, epoch)
    writer.add_scalar('Validation/Recall', val_recall, epoch)
    writer.add_scalar('Validation/F1 Score', val_f1, epoch)
    
    if val_loss/len(val_loader) < best_val_loss:
        best_val_loss = val_loss/len(val_loader)
        torch.save(model.state_dict(), config['paths']['model_save_path'])
        
writer.close()