import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from aggregation import get_aggregation_stratey
from model_loader import get_model_hidden_size, get_raw_features
import numpy as np

def save_checkpoint(path, classifier, optimizer, epoch, history):
    checkpoint = {
        'classifier_state': classifier.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'history': history
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, classifier, optimizer):
    checkpoint = torch.load(path, weights_only=False)
    classifier.load_state_dict(checkpoint['classifier_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    return classifier, optimizer, epoch, history

def load_history_from_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    return checkpoint.get('history', None)

def train_model(
    model,
    dataset_name,
    strategy,
    trial_number,
    train_loader,
    val_loader,
    test_loader,
    num_epochs,
    learning_rate,
    optimizer_type,
    momentum,
    weight_decay,
    scheduler_type,
    checkpoint_root_path,
    load_checkpoint_flag
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this implementation")

    num_classes = train_loader.dataset.num_classes
    classifier = nn.Linear(get_model_hidden_size(model), num_classes).cuda()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    criterion = nn.CrossEntropyLoss()
    model = model.cuda()
    model.eval()
    history = {
        'val_acc': [],
        'test_acc': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': []
    }
    start_epoch = 0
    
    if not os.path.exists(f"{checkpoint_root_path}/checkpoints"):
        os.mkdir(f"{checkpoint_root_path}/checkpoints")

    checkpoint_path = f'{checkpoint_root_path}/checkpoints/{model.config.name_or_path.replace("/", "_")}_{strategy}_{dataset_name}_{trial_number}.pt'

    if load_checkpoint_flag and os.path.exists(checkpoint_path):
        classifier, optimizer, start_epoch, history = load_checkpoint(checkpoint_path, classifier, optimizer)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    for epoch in range(start_epoch, num_epochs):

        # Train
        classifier.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(inputs)
                features = get_raw_features(model, outputs)
                features = get_aggregation_stratey(strategy)(features)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if scheduler:
            scheduler.step()

        history['train_loss'].append(np.mean(train_losses))

        # Validation
        classifier.eval()
        val_preds, val_labels, val_losses = [], [], []
        for inputs, labels in tqdm(val_loader, desc='Evaluating (val)', leave=False):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
                features = get_raw_features(model, outputs)
                features = get_aggregation_stratey(strategy)(features)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            val_losses.append(loss.item()) 
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_acc = (np.array(val_preds) == np.array(val_labels)).mean() * 100
        history['val_acc'].append(val_acc)
        history['val_loss'].append(np.mean(val_losses))

        # Test
        test_preds, test_labels, test_losses = [], [], []
        for inputs, labels in tqdm(test_loader, desc='Evaluating (test)', leave=False):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(inputs)
                features = get_raw_features(model, outputs)
                features = get_aggregation_stratey(strategy)(features)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        test_acc = (np.array(test_preds) == np.array(test_labels)).mean() * 100
        history['test_acc'].append(test_acc)
        history['test_loss'].append(np.mean(test_losses))

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'Train - Loss: {history['train_loss'][-1]:.4f}')
        print(f'Val   - Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.2f}%')
        print(f'Test  - Loss: {history['test_loss'][-1]:.4f}, Acc: {history['test_acc'][-1]:.2f}%')

        save_checkpoint(checkpoint_path, classifier, optimizer, epoch + 1, history)

    return history