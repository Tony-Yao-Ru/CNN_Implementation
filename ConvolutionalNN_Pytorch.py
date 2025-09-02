# ConvolutionalNN_Pytorch.py
import torch
from torch import nn
import matplotlib.pyplot as plt

class ConvolutionalNN_Pytorch(nn.Module):
    def __init__(
        self,
        input_channels=3,
        conv_channels=[32, 64],
        conv_dim=2,
        kernel_size=3,
        pool_size=2,
        num_classes=10,
        hidden_units=[256],
        use_padding=True,
        dropout=0.2,
        device=None,
        *,
        pool_every=1,           # build-time pooling frequency
        conv_stride=1           # build-time conv stride
    ):
        """
        General N-D CNN with GAP head.
        pool_every/conv_stride are applied at build time. (The Train() method
        also accepts these names but only to stay API-compatible with the NumPy version.)
        """
        super().__init__()
        self.conv_dim = conv_dim
        self.num_classes = num_classes

        # --- dimension-specific ops ---
        if conv_dim == 1:
            Conv, Pool, BN, GAP = nn.Conv1d, nn.MaxPool1d, nn.BatchNorm1d, nn.AdaptiveAvgPool1d
            gap_out = 1
        elif conv_dim == 2:
            Conv, Pool, BN, GAP = nn.Conv2d, nn.MaxPool2d, nn.BatchNorm2d, nn.AdaptiveAvgPool2d
            gap_out = (1, 1)
        elif conv_dim == 3:
            Conv, Pool, BN, GAP = nn.Conv3d, nn.MaxPool3d, nn.BatchNorm3d, nn.AdaptiveAvgPool3d
            gap_out = (1, 1, 1)
        else:
            raise ValueError("conv_dim must be 1, 2, or 3")

        padding = (kernel_size // 2) if (use_padding and kernel_size % 2 == 1) else 0

        # --- conv stack with optional pooling frequency ---
        conv_layers = []
        in_ch = input_channels
        k = int(pool_every) if pool_every else 0
        for idx, out_ch in enumerate(conv_channels):
            conv_layers += [
                Conv(in_ch, out_ch, kernel_size=kernel_size, stride=conv_stride, padding=padding, bias=False),
                BN(out_ch),
                nn.ReLU(inplace=True),
            ]
            if k and ((idx + 1) % k == 0):
                conv_layers += [Pool(pool_size)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # --- GAP + FC head (shape-agnostic thanks to GAP) ---
        self.gap = GAP(gap_out)
        self.flatten = nn.Flatten()
        feat_dim = conv_channels[-1] if len(conv_channels) > 0 else input_channels

        fc_layers = []
        prev = feat_dim
        for h in (hidden_units or []):
            fc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout and dropout > 0:
                fc_layers += [nn.Dropout(dropout)]
            prev = h
        fc_layers += [nn.Linear(prev, num_classes)]
        self.fc = nn.Sequential(*fc_layers)

        # --- runtime/device ---
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # ---------- forward ----------
    def forward(self, x):
        x = self.conv(x) if len(self.conv) > 0 else x
        x = self.gap(x)
        x = self.flatten(x)   # [B, C]
        x = self.fc(x)
        return x

    # ---------- loss + metrics ----------
    def _loss_and_metrics(self, logits, y, loss_fn):
        if self.num_classes == 1:
            logits_flat = logits.squeeze(1)      # [B]
            loss = loss_fn(logits_flat, y.float())
            probs = torch.sigmoid(logits_flat)
            preds = (probs >= 0.5).long()
        else:
            loss = loss_fn(logits, y.long())
            preds = logits.argmax(dim=1)
        correct = (preds == y.long()).sum().item()
        return loss, correct

    # ---------- one epoch train ----------
    def train_loop(self, loader, optimizer, loss_fn):
        self.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = self(X)
            loss, corr = self._loss_and_metrics(logits, y, loss_fn)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            correct += corr
            total += bs
        return running_loss / max(total, 1), 100.0 * correct / max(total, 1)

    # ---------- evaluation ----------
    @torch.no_grad()
    def test_loop(self, loader, loss_fn):
        self.eval()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            logits = self(X)
            loss, corr = self._loss_and_metrics(logits, y, loss_fn)
            bs = y.size(0)
            running_loss += loss.item() * bs
            correct += corr
            total += bs
        return running_loss / max(total, 1), 100.0 * correct / max(total, 1)

    # ---------- Train (accepts extra conv args for runner compatibility) ----------
    def Train(self,
              train_loader,
              val_loader=None,
              epochs=10,
              optimizer_name="adam",
              lr=1e-3,
              beta=0.9,
              beta1=0.9,
              beta2=0.999,
              eps=1e-8,
              weight_decay=0.0,
              print_every=1,
              # The following args are accepted to match the NumPy API / your runner,
              # but are NOT used here (conv stack already built in __init__):
              pool_every=None,
              conv_stride=None,
              conv_padding=None):
        """
        Note: pool_every/conv_stride/conv_padding are ignored at train time for PyTorch.
        Set them in __init__ when constructing the model if you want them to take effect.
        """
        loss_fn = nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss()

        opt_name = optimizer_name.lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "momentum":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=beta, nesterov=False, weight_decay=weight_decay)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))

        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = self.train_loop(train_loader, optimizer, loss_fn)
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)

            if val_loader is not None:
                val_loss, val_acc = self.test_loop(val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss = val_acc = None

            ax.cla()
            ax.set_title("Training & Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.plot(history["train_loss"], label="Train Loss")
            if val_loader is not None:
                ax.plot(history["val_loss"], label="Val Loss")
            ax.legend()
            plt.pause(0.1)

            if (epoch % print_every) == 0:
                msg = f"Epoch {epoch:02d} | train loss {tr_loss:.4f}, acc {tr_acc:5.1f}%"
                if val_loss is not None:
                    msg += f" | val loss {val_loss:.4f}, acc {val_acc:5.1f}%"
                print(msg)

        plt.ioff()
        plt.show()
        return history

    @torch.no_grad()
    def predict(self, x, return_prob=False, threshold=0.5):
        """
        Predict for a tensor, numpy array, or batch. Handles binary & multi-class.
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        expected_input_dims = 1 + self.conv_dim  # C + spatial dims
        if x.dim() == expected_input_dims:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        logits = self(x)

        if self.num_classes == 1:
            probs = torch.sigmoid(logits.squeeze(1))
            preds = (probs >= threshold).long().cpu()
            return (preds, probs.cpu()) if return_prob else preds
        else:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu()
            return (preds, probs.cpu()) if return_prob else preds
