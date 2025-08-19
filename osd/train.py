# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from dataset import OverlapSpeechDataset
# from model import CRNN
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# def calculate_accuracy(preds, labels):
#     preds = (preds > 0.5).float()
#     correct = (preds == labels).float().sum()
#     accuracy = correct / labels.numel()
#     return accuracy.item()

# def collate_fn(batch):
#     features, labels = zip(*batch)
#     lengths = [f.shape[0] for f in features]
#     max_len = max(lengths)
#     feats = torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features])
#     labs = torch.stack([torch.nn.functional.pad(l, (0, max_len - len(l))) for l in labels])
#     return feats.to(device), labs.to(device)

# # Dataset and splits
# train_dataset = OverlapSpeechDataset("data/synthetic_split/train/audio", "data/synthetic_split/train/labels")
# val_dataset = OverlapSpeechDataset("data/synthetic_split/val/audio", "data/synthetic_split/val/labels")

# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# # Model setup
# model = CRNN().to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)

# best_val_accuracy = 0.0

# for epoch in range(20):
#     # Training phase
#     model.train()
#     train_loss, train_acc = 0.0, 0.0
#     for features, labels in train_loader:
#         optimizer.zero_grad()
#         preds = model(features)
#         loss = criterion(preds, labels)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
#         train_acc += calculate_accuracy(preds, labels)
    
#     # Validation phase
#     model.eval()
#     val_loss, val_acc = 0.0, 0.0
#     with torch.no_grad():
#         for features, labels in val_loader:
#             preds = model(features)
#             val_loss += criterion(preds, labels).item()
#             val_acc += calculate_accuracy(preds, labels)
    
#     # Calculate averages
#     train_loss /= len(train_loader)
#     train_acc /= len(train_loader)
#     val_loss /= len(val_loader)
#     val_acc /= len(val_loader)
    
#     # Print statistics
#     print(f"\nEpoch {epoch+1}:")
#     print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
#     print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
#     # Save best model
#     if val_acc > best_val_accuracy:
#         best_val_accuracy = val_acc
#         torch.save(model.state_dict(), 'model_osd.pth')
#         print("  Saved new best model")

# # After training, load best model for testing/inference
# model.load_state_dict(torch.load('model_osd.pth'))


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from dataset import OverlapSpeechDataset
# from model import CRNN
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# def calculate_accuracy(preds, labels):
#     preds = (preds > 0.5).float()
#     correct = (preds == labels).float().sum()
#     accuracy = correct / labels.numel()
#     return accuracy.item()

# def collate_fn(batch):
#     features, labels = zip(*batch)
#     lengths = [f.shape[0] for f in features]
#     max_len = max(lengths)
#     feats = torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features])
#     labs = torch.stack([torch.nn.functional.pad(l, (0, max_len - len(l))) for l in labels])
#     return feats.to(device), labs.to(device)

# # Dataset and splits
# train_dataset = OverlapSpeechDataset("data/synthetic_split/train/audio", "data/synthetic_split/train/labels")
# val_dataset = OverlapSpeechDataset("data/synthetic_split/val/audio", "data/synthetic_split/val/labels")

# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# # Model setup
# model = CRNN().to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# best_val_accuracy = 0.0

# for epoch in range(20):
#     # Training phase
#     model.train()
#     train_loss, train_acc = 0.0, 0.0
#     for features, labels in train_loader:
#         optimizer.zero_grad()
#         preds = model(features)
#         loss = criterion(preds, labels)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
#         train_acc += calculate_accuracy(preds, labels)
    
#     # Validation phase
#     model.eval()
#     val_loss, val_acc = 0.0, 0.0
#     with torch.no_grad():
#         for features, labels in val_loader:
#             preds = model(features)
#             val_loss += criterion(preds, labels).item()
#             val_acc += calculate_accuracy(preds, labels)
    
#     # Calculate averages
#     train_loss /= len(train_loader)
#     train_acc /= len(train_loader)
#     val_loss /= len(val_loader)
#     val_acc /= len(val_loader)
    
#     # Update learning rate based on validation loss
#     scheduler.step(val_loss)
    
#     # Print statistics
#     print(f"\nEpoch {epoch+1}:")
#     print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
#     print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
#     print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
    
#     # Save best model
#     if val_acc > best_val_accuracy:
#         best_val_accuracy = val_acc
#         torch.save(model.state_dict(), 'model_osd.pth')
#         print("  Saved new best model")

# # After training, load best model for testing/inference
# model.load_state_dict(torch.load('model_osd.pth'))
# print("Training complete. Best model loaded for inference.")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OverlapSpeechDataset
from model import CRNN
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# def calculate_accuracy(preds, labels):
#     preds = (preds > 0.5).float()
#     correct = (preds == labels).float().sum()
#     accuracy = correct / labels.numel()
#     return accuracy.item()

def calculate_accuracy(logits, labels):
    probs = torch.sigmoid(logits)           # Convert logits to probabilities
    preds = (probs > 0.5).float()            # Threshold at 0.5
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.numel()
    return accuracy.item()

def collate_fn(batch):
    features, labels = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feats = torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in features])
    labs = torch.stack([torch.nn.functional.pad(l, (0, max_len - len(l))) for l in labels])
    return feats.to(device), labs.to(device)

# Dataset and splits
train_dataset = OverlapSpeechDataset("data/synthetic_split/train/audio", "data/synthetic_split/train/labels")
val_dataset = OverlapSpeechDataset("data/synthetic_split/val/audio", "data/synthetic_split/val/labels")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Model setup
model = CRNN().to(device)
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()  # ✅ Use logits-based loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

best_val_accuracy = 0.0

for epoch in range(20):
    # Training phase
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        logits = model(features)  # Raw outputs without sigmoid
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += calculate_accuracy(logits, labels)
        # preds = model(features)
        # loss = criterion(preds, labels)
        # loss.backward()
        # optimizer.step()
        
        # train_loss += loss.item()
        # train_acc += calculate_accuracy(preds, labels)
    
    # Validation phase
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            logits = model(features)
            val_loss += criterion(logits, labels).item()
            val_acc += calculate_accuracy(logits, labels)
            # preds = model(features)
            # val_loss += criterion(preds, labels).item()
            # val_acc += calculate_accuracy(preds, labels)
    
    # Calculate averages
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    scheduler.step(val_loss)
    
    # Print statistics
    print(f"\nEpoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'CRNN_OSD_MODEL.pth')
        print("  Saved new best model")

# After training, load best model for testing/inference
model.load_state_dict(torch.load('CRNN_OSD_MODEL.pth'))