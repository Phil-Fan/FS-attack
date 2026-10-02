
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = TensorDataset = None

    class _MissingTorchModule:
        pass

    class _MissingTorchNN:
        Module = _MissingTorchModule

    nn = _MissingTorchNN()
    logger.warning("Torch not installed, the tsai backend is unavailable.")


def get_tsai_config(num_classes: int = 10, model_type: str = "minirocket") -> Dict:
    if not TORCH_AVAILABLE:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "model_type": model_type,
        "device": device,
        "num_classes": num_classes,
        "num_kernels": 1000,
        "epochs": 30,
    }

    if model_type == "minirocket":
        config.update({
            "num_kernels": 10000,
            "max_dilations": 32,
            "verbose": False,
        })
    elif model_type == "inceptiontime":
        config.update({
            "in_channels": 1,
            "base_filters": 32,
            "kernel_size": 41,
            "stride": 1,
            "depth": 6,
            "dropout": 0.2,
        })

    return config


def parse_itl_sequence(itl_str: str) -> np.ndarray:
    if isinstance(itl_str, str):
        return np.array([float(x) for x in itl_str.split(',')])
    return np.array(itl_str)


def prepare_sequence_data(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label_col: str = 'kv_label',
    seq_col: str = 'itl'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_seqs = [parse_itl_sequence(s) for s in train_df[seq_col].values]
    eval_seqs = [parse_itl_sequence(s) for s in eval_df[seq_col].values]

    max_len = max(max(len(s) for s in train_seqs), max(len(s) for s in eval_seqs))
    logger.info(f"Maximum sequence length: {max_len}")

    def pad_sequences(seqs, target_len):
        padded = []
        for seq in seqs:
            if len(seq) < target_len:
                seq = np.pad(seq, (0, target_len - len(seq)), mode='constant', constant_values=0)
            elif len(seq) > target_len:
                seq = seq[:target_len]
            padded.append(seq)
        return np.array(padded)

    X_train = pad_sequences(train_seqs, max_len)
    X_eval = pad_sequences(eval_seqs, max_len)

    y_train = train_df[label_col].values.astype(int)
    y_eval = eval_df[label_col].values.astype(int)

    min_label = min(y_train.min(), y_eval.min())
    y_train = y_train - min_label
    y_eval = y_eval - min_label

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_eval = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[1])

    logger.info(f"Sequence data prepared: X_train={X_train.shape}, X_eval={X_eval.shape}")

    return X_train, y_train, X_eval, y_eval


class MiniRocketPyTorch(nn.Module):

    def __init__(self, input_length, num_classes=10, num_kernels=10000, max_dilations=32):
        super().__init__()
        self.input_length = input_length
        self.num_kernels = num_kernels

        np.random.seed(42)
        self._init_kernels(input_length, max_dilations)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_kernels * 3, num_classes)

    def _init_kernels(self, input_length, max_dilations):
        dilations = []
        d = 1
        while d < min(input_length, max_dilations):
            dilations.append(d)
            d *= 2
        if len(dilations) == 0:
            dilations = [1]

        kernels_per_dilation = self.num_kernels // len(dilations)

        dilations_list = []
        weights_list = []

        for dilation in dilations:
            n_kernels = kernels_per_dilation
            if dilation == dilations[-1]:
                n_kernels = self.num_kernels - len(dilations_list)

            kernel_size = min(9, input_length // dilation if dilation > 0 else input_length)
            if kernel_size < 1:
                kernel_size = 1

            for _ in range(n_kernels):
                dilations_list.append(dilation)
                weights_list.append(torch.randn(1, 1, kernel_size) * 0.01)

        self.dilations = dilations_list
        max_kernel_size = max(w.shape[2] for w in weights_list)
        self.max_kernel_size = max_kernel_size
        self.kernel_sizes = [w.shape[2] for w in weights_list]

        stacked_weights = torch.zeros(self.num_kernels, 1, max_kernel_size)
        for i, w in enumerate(weights_list):
            stacked_weights[i, 0, :w.shape[2]] = w.squeeze()

        self.register_buffer('stacked_weights', stacked_weights)

        logger.info(f"MiniRocket: {self.num_kernels} kernels, {len(dilations)} dilation groups")

    def forward(self, x):
        batch_size = x.shape[0]

        all_features = []

        for dilation in set(self.dilations):
            indices = [i for i, d in enumerate(self.dilations) if d == dilation]
            n_kernels = len(indices)

            if n_kernels == 0:
                continue

            kernel_size = self.kernel_sizes[indices[0]]

            for idx in indices:
                weight = self.stacked_weights[idx:idx+1, :, :kernel_size]
                weight = weight.to(x.device)

                out = torch.nn.functional.conv1d(
                    x, weight, bias=None, dilation=dilation, padding='same'
                )

                ppv = (out > 0).float().mean(dim=2)
                nnv = (out < 0).float().mean(dim=2)
                mean = out.mean(dim=2)

                all_features.append(ppv)
                all_features.append(nnv)
                all_features.append(mean)

        x = torch.cat(all_features, dim=1)
        x = self.fc(x)
        return x

        x = self.fc(x)
        return x


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()

        self.kernel_size = kernel_size

        self.branch1 = nn.Conv1d(in_channels, out_channels, 1, padding='same')

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        )

        self.pool = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.MaxPool1d(3, padding=1, stride=1)
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.pool(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out


class InceptionTimePyTorch(nn.Module):

    def __init__(self, input_length, num_classes=10, in_channels=1,
                 base_filters=32, kernel_size=41, depth=6, dropout=0.2):
        super().__init__()

        self.input_length = input_length
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters * 4, 1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU()
        )

        self.inception_blocks = nn.ModuleList()
        for i in range(depth):
            block = InceptionBlock(
                in_channels=base_filters * 4,
                out_channels=base_filters,
                kernel_size=kernel_size,
                dropout=dropout
            )
            self.inception_blocks.append(block)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 4, num_classes)

    def forward(self, x):

        x = self.stem(x)

        residual = x
        for i, block in enumerate(self.inception_blocks):
            x = block(x)

            if i % 3 == 0 and x.shape != residual.shape:
                residual = torch.nn.functional.interpolate(
                    residual, size=x.shape[2], mode='linear', align_corners=False
                )
            if x.shape == residual.shape:
                x = x + residual
                residual = x

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TsaiBackend:

    @property
    def backend_name(self) -> str:
        return f"tsai_{self.model_type}"

    def __init__(
        self,
        config: Optional[Dict] = None,
        num_classes: int = 10,
        model_type: str = "minirocket"
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")

        self.model_type = model_type
        self.num_classes = num_classes
        self.config = config or get_tsai_config(num_classes, model_type)
        self.model = None
        self.train_acc = None
        self.input_length = None
        self.device = torch.device(self.config.get("device", "cpu"))

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        logger.info(f"Training {self.model_type} classifier...")
        logger.info(f"X_train shape: {X_train.shape}, num_classes: {self.num_classes}")

        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        self.input_length = X_train.shape[2]

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train.astype(int))
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        if self.model_type == "minirocket":
            self.model = MiniRocketPyTorch(
                input_length=self.input_length,
                num_classes=self.num_classes,
                num_kernels=self.config.get("num_kernels", 10000)
            )
        elif self.model_type == "inceptiontime":
            self.model = InceptionTimePyTorch(
                input_length=self.input_length,
                num_classes=self.num_classes,
                in_channels=self.config.get("in_channels", 1),
                base_filters=self.config.get("base_filters", 32),
                kernel_size=self.config.get("kernel_size", 40),
                depth=self.config.get("depth", 6),
                dropout=self.config.get("dropout", 0.2)
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.model = self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        epochs = self.config.get("epochs", 50)
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += predicted.size(0)
                correct += predicted.eq(batch_y).sum().item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={100.*correct/total:.2f}%")

        self.model.eval()
        with torch.no_grad():
            X_train_t = X_train_t.to(self.device)
            outputs = self.model(X_train_t)
            _, y_pred = outputs.max(1)
            self.train_acc = (y_pred.cpu() == y_train_t).float().mean().item()

        logger.info(f"Train accuracy: {self.train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predictions = outputs.max(1)

        return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        X_t = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def save(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), model_dir / 'model.pt')

        config_data = {
            'backend': self.backend_name,
            'model_type': self.model_type,
            'config': self.config,
            'num_classes': self.num_classes,
            'input_length': self.input_length,
            'train_acc': self.train_acc,
        }
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Model saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        model_dir = Path(model_dir)

        with open(model_dir / 'config.json', 'r') as f:
            config_data = json.load(f)

        self.model_type = config_data['model_type']
        self.config = config_data['config']
        self.num_classes = config_data['num_classes']
        self.input_length = config_data.get('input_length')
        self.train_acc = config_data.get('train_acc')
        self.device = torch.device(self.config.get("device", "cpu"))

        if self.model_type == "minirocket":
            self.model = MiniRocketPyTorch(
                input_length=self.input_length,
                num_classes=self.num_classes,
                num_kernels=self.config.get("num_kernels", 10000)
            )
        elif self.model_type == "inceptiontime":
            self.model = InceptionTimePyTorch(
                input_length=self.input_length,
                num_classes=self.num_classes,
                in_channels=self.config.get("in_channels", 1),
                base_filters=self.config.get("base_filters", 32),
                kernel_size=self.config.get("kernel_size", 40),
                depth=self.config.get("depth", 6),
                dropout=self.config.get("dropout", 0.2)
            )

        self.model.load_state_dict(torch.load(model_dir / 'model.pt', weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded from {model_dir}")
