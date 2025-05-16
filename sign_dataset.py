import os
import numpy as np
from torch.utils.data import Dataset
import torch

class SignPoseDataset(Dataset):
    def __init__(self, root_dir, max_frames=60):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.max_frames = max_frames

        class_folders = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_folders):
            self.label_map[class_name] = idx
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(class_path, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose_seq = np.load(self.data[idx])  # shape: (60, 225)
        label = self.labels[idx]

        pose_tensor = torch.tensor(pose_seq, dtype=torch.float32)  # [60, 225]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return pose_tensor, label_tensor