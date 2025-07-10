import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Dataset path
data_path = '/mnt/c/Users/throw/OneDrive/Desktop/asl_alphabet_train'

# ✅ Image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ Load dataset
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# ✅ 70/15/15 split
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

print(f"Classes: {full_dataset.classes}")
print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

# ✅ CNN model with AdaptiveAvgPool2d (safe for any input size)
class ASLClassifier(nn.Module):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ✅ Initialize model
model = ASLClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ Early stopping
best_val_loss = float('inf')
patience = 3
epochs_no_improve = 0
early_stop = False
epochs = 20

# ✅ Training loop with tqdm
for epoch in range(epochs):
    if early_stop:
        print("🔁 Early stopping triggered.")
        break

    model.train()
    running_loss = 0.0
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_progress.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # ✅ Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1} Summary → Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("✅ Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            early_stop = True

# ✅ Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Final Test Accuracy: {accuracy * 100:.2f}%")
