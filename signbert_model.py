import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import mediapipe as mp
import cv2

class SignBERT(nn.Module):
    def __init__(self, input_dim=225, num_classes=10, num_layers=2, num_heads=4,
                 hidden_dim=256, dropout=0.1, max_seq_len=60):
        super(SignBERT, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len + 1  # +1 for CLS token

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, 0, :]
        return self.classifier(x)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def evaluate_confusion_matrix(model, data_dir, label_map):
    model.eval()
    y_true = []
    y_pred = []

    for label, class_name in label_map.items():
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".npy"):
                npy_path = os.path.join(class_dir, file_name)
                data = np.load(npy_path)
                data = torch.tensor([data], dtype=torch.float32)

                with torch.no_grad():
                    output = model(data)
                    pred = torch.argmax(output, dim=1).item()

                y_true.append(label)
                y_pred.append(pred)

    plot_confusion_matrix(y_true, y_pred, [label_map[i] for i in range(len(label_map))])
