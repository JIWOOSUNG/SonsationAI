import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sign_dataset import SignPoseDataset
from signbert_model import SignBERT

# 학습 설정
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
DATA_DIR = "pose_npy_normalized_filtered"  # npy와 라디마 구조
MODEL_SAVE_PATH = "signbert_model.pth"
MAX_FRAMES = 100  # 프레임 수정!

# 데이터셋 준비
dataset = SignPoseDataset(DATA_DIR, max_frames=MAX_FRAMES)
num_classes = len(dataset.label_map)
dataset.label_map_inv = {v: k for k, v in dataset.label_map.items()}
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 클래스 개수 비급성 해결효과
labels_all = [label for _, label in train_dataset]
class_sample_counts = np.bincount(labels_all)
class_weights = 1. / (class_sample_counts + 1e-6)
weights = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignBERT(input_dim=225, num_classes=num_classes, max_seq_len=MAX_FRAMES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 로그 저장용 리스트
train_losses = []
val_accuracies = []
best_acc = 0

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    train_losses.append(total_loss)
    val_accuracies.append(acc)

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Val Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_map': dataset.label_map
        }, MODEL_SAVE_PATH)
        print(f"✅ 모델 저장 (Best Accuracy: {best_acc:.2f}%)")

# 학습 그래프 시각화
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.tight_layout()
plt.show()

# PCA 시각화
print("프레임 프리점 시각화...")
model.eval()
features = []
labels = []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        feats = model(x)
        features.append(feats.cpu().numpy())
        labels.extend(y.numpy())

features = np.concatenate(features, axis=0)
labels = np.array(labels)

pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    idx = labels == label
    label_name = dataset.label_map_inv[label]
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label_name, s=40)
plt.title("PCA of Output Features (SignBERT)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

scripted_model = torch.jit.script(model)
scripted_model.save("signbert_model_scripted.pt")