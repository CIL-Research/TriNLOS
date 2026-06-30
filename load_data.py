import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from scipy.io import loadmat


def random_flip_and_rotate(volume, intensity, depthmap):
    flip_ud = random.random() < 0.5
    flip_lr = random.random() < 0.5
    k = random.choice([0, 1, 2, 3])

    if flip_ud:
        volume = torch.flip(volume, dims=[-1])
        intensity = torch.flip(intensity, dims=[-1])
        depthmap = torch.flip(depthmap, dims=[-1])

    if flip_lr:
        volume = torch.flip(volume, dims=[-2])
        intensity = torch.flip(intensity, dims=[-2])
        depthmap = torch.flip(depthmap, dims=[-2])

    if k > 0:
        volume = torch.rot90(volume, k=k, dims=[-2, -1])
        intensity = torch.rot90(intensity, k=k, dims=[-2, -1])
        depthmap = torch.rot90(depthmap, k=k, dims=[-2, -1])

    return volume, intensity, depthmap


def load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hdr":
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img if img.ndim == 2 else np.tensordot(img, [0.114, 0.587, 0.299], axes=([2], [0]))
        if img is None:
            raise FileNotFoundError(f"Cannot read HDR image: {path}")
        return img.astype(np.float32)

    elif ext == ".mat":
        mat = loadmat(path)
        if "img" in mat:
            arr = mat["img"][0][:, :, ]
            arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
        elif "depth" in mat:
            arr = mat["depth"][0][:, :, 0]
        else:
            raise KeyError(f"Neither 'img' nor 'depth' found in MAT file: {path}")
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Loaded array from {path} has shape {arr.shape}, expected 2D.")
        return arr.astype(np.float32)

    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .hdr or .mat")


def mp42arr(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.array(frames)[:512, :, :]


def average_pooling(tensor, down_factor):
    return F.avg_pool3d(
        tensor,
        kernel_size=(down_factor, down_factor, down_factor),
        stride=(down_factor, down_factor, down_factor),
    )


def average_spapooling(tensor, down_factor=2):
    return F.avg_pool3d(
        tensor,
        kernel_size=(1, down_factor, down_factor),
        stride=(1, down_factor, down_factor),
    )


def average_pooling2d(tensor, down_factor):
    return F.avg_pool2d(tensor, kernel_size=down_factor, stride=down_factor)


class Renderdataset(Dataset):
    def __init__(self, data_frame, down_value, train):
        self.nlos_data_list = data_frame["nlos_path"]
        self.img_data_list = data_frame["img_path"]
        self.dep_data_list = data_frame["dep_path"]
        self.down_value = down_value
        self.training = train

    def __len__(self):
        return len(self.nlos_data_list)

    def __getitem__(self, idx):
        nlos_path = self.nlos_data_list[idx]
        img_path = self.img_data_list[idx]
        dep_path = self.dep_data_list[idx]

        nlos_ar = torch.tensor(mp42arr(nlos_path) / 255, dtype=torch.float32).unsqueeze(0)
        img = torch.tensor(load_image(img_path), dtype=torch.float32).unsqueeze(0)
        dep = torch.tensor(load_image(dep_path), dtype=torch.float32).unsqueeze(0)

        if self.down_value > 1:
            img = average_pooling2d(img, self.down_value)
            dep = average_pooling2d(dep, self.down_value)

        img = (img - img.min()) / (img.max() - img.min())
        nlos_ar = nlos_ar / nlos_ar.max()

        if self.training:
            nlos_ar, img, dep = random_flip_and_rotate(nlos_ar, img, dep)

        return nlos_ar, img, dep
