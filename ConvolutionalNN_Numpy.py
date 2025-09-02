import numpy as np
import matplotlib.pyplot as plt

class ConvolutionalNN_Numpy:
    def __init__(self,
        input_channels=3,
        conv_channels=[32, 64],
        conv_dim=2,
        kernel_size=3,
        pool_size=2,
        num_classes=10,
        hidden_units=[256],
        use_padding=True,
        dropout=0.2,
        device=None
    ):
        """
        NumPy implementation of an N-D CNN with optional FC head.
        Supports:
          - conv_dim in {1,2,3}
          - MaxPool
          - ReLU
          - Softmax + CrossEntropy (multi-class)
          - Sigmoid + BCE-with-logits (binary when num_classes==1)
          - SGD or Adam optimizer
        """
        super().__init__()
        if conv_channels is None: conv_channels = [32, 64]
        if hidden_units  is None: hidden_units  = [256]

        self.conv_dim = conv_dim
        self.use_padding = use_padding
        self.pool_size = (pool_size,) * conv_dim if isinstance(pool_size, int) else tuple(pool_size)
        self.kernel_size = (kernel_size,) * conv_dim if isinstance(kernel_size, int) else tuple(kernel_size)

        self.weighted_vectors = {}
        self.bias_vectors     = {}
        self.gradients        = {}

        in_c = input_channels
        kshape = self.kernel_size
        for i, out_c in enumerate(conv_channels):
            fan_in = in_c * np.prod(kshape)
            W = (np.random.randn(out_c, in_c, *kshape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
            b = np.zeros((out_c, 1), dtype=np.float32)
            self.weighted_vectors[f"W_conv{i+1}"] = W
            self.bias_vectors[f"b_conv{i+1}"]     = b
            self.gradients[f"dW_conv{i+1}"]       = np.zeros_like(W)
            self.gradients[f"db_conv{i+1}"]       = np.zeros_like(b)
            self.weighted_vectors[f"META_conv{i+1}"] = {
                "stride": (1,) * conv_dim,
                "padding": (1,) * conv_dim if use_padding else (0,) * conv_dim,
                "kernel_size": kshape,
            }
            in_c = out_c

        self.conv_layers      = len(conv_channels)
        self.feature_channels = in_c

        self.hidden_units = list(hidden_units)
        self.num_classes  = num_classes
        self.fc_built     = False
        self.fc_layers    = 0
        self._opt_state   = {}

    # ------------------ optimizer step ------------------
    def train_loop(self, lr=1e-3, optimizer="adam", beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        def _param_names():
            for name in list(self.weighted_vectors.keys()):
                if name.startswith("W_"):
                    yield name
            for name in list(self.bias_vectors.keys()):
                if name.startswith("b_"):
                    yield name

        for pname in _param_names():
            gname = "d" + pname
            if gname not in self.gradients:
                continue
            if pname.startswith("W_"):
                P = self.weighted_vectors[pname]
                G = self.gradients[gname]
                if weight_decay > 0.0:
                    G = G + weight_decay * P
            else:
                P = self.bias_vectors[pname]
                G = self.gradients[gname]

            if optimizer.lower() == "sgd":
                P -= lr * G
            else:  # Adam
                st = self._opt_state.setdefault(pname, {"t":0, "m":np.zeros_like(P), "v":np.zeros_like(P)})
                st["t"] += 1
                st["m"]  = beta1 * st["m"] + (1 - beta1) * G
                st["v"]  = beta2 * st["v"] + (1 - beta2) * (G * G)
                m_hat = st["m"] / (1 - beta1**st["t"])
                v_hat = st["v"] / (1 - beta2**st["t"])
                P -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if pname.startswith("W_"):
                self.weighted_vectors[pname] = P
            else:
                self.bias_vectors[pname] = P

    # ------------------ training loop -------------------
    def Train(self, train_loader, val_loader=None, epochs=10, optimizer_name="adam",
          lr=1e-3, beta=0.9, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
          print_every=1, pool_every=1, conv_stride=1, conv_padding=None):
        
        def to_onehot(y, C):
            y = y.astype(int)
            oh = np.zeros((y.shape[0], C), dtype=np.float32)
            oh[np.arange(y.shape[0]), y] = 1.0
            return oh

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Live plot: TRAIN + VAL LOSS
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Training & Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True)
        train_line, = ax.plot([], [], label="Train Loss")
        val_line,   = ax.plot([], [], label="Val Loss")
        ax.legend()

        for epoch in range(1, epochs+1):
            # -------- train one epoch --------
            total_loss = total_acc = 0.0
            total_n    = 0
            for X, y in train_loader:
                X = X.astype(np.float32)
                y = y.astype(np.int64)

                # labels:
                if self.num_classes == 1:
                    y1h = y.reshape(-1, 1).astype(np.float32)  # (N,1)
                else:
                    y1h = to_onehot(y, self.num_classes)       # (N,C)

                loss, probs = self.Backward_Propagation(
                    X, y1h,
                    pool_every=pool_every,
                    conv_stride=conv_stride,
                    conv_padding=conv_padding
                )

                # optimizer step
                self.train_loop(lr=lr, optimizer=optimizer_name,
                                beta1=beta1, beta2=beta2, eps=eps,
                                weight_decay=weight_decay)

                # accuracy
                if self.num_classes == 1:
                    pred = (probs.reshape(-1) >= 0.5).astype(np.int64)
                else:
                    pred = np.argmax(probs, axis=1)

                acc  = np.mean(pred == y)
                bs   = X.shape[0]
                total_loss += float(loss) * bs
                total_acc  += float(acc)  * bs
                total_n    += bs

            train_loss_epoch = total_loss / max(total_n, 1)
            train_acc_epoch  = total_acc  / max(total_n, 1)
            history["train_loss"].append(train_loss_epoch)
            history["train_acc"].append(train_acc_epoch)

            # -------- validation (LOSS + ACC) --------
            if val_loader is not None:
                vl = va = 0.0
                vn = 0
                for Xv, yv in val_loader:
                    Xv = Xv.astype(np.float32)
                    logits, _ = self.Forward_Propagation(
                        Xv,
                        pool_every=pool_every,
                        conv_stride=conv_stride,
                        conv_padding=conv_padding
                    )
                    if self.num_classes == 1:
                        pv    = self.sigmoid(logits)                         # (N,1)
                        yv1h  = yv.reshape(-1,1).astype(np.float32)
                        vloss = self.bce_with_logits_loss(yv1h, logits)
                        vpred = (pv >= 0.5).astype(np.int64).reshape(-1)
                    else:
                        pv    = self.softmax(logits)
                        yv1h  = np.eye(self.num_classes, dtype=np.float32)[yv.astype(int)]
                        vloss = self.cross_entropy_loss(yv1h, pv)
                        vpred = np.argmax(pv, axis=1)
                    vacc  = np.mean(vpred == yv)

                    bs = Xv.shape[0]
                    vl += float(vloss) * bs
                    va += float(vacc)  * bs
                    vn += bs

                history["val_loss"].append(vl / max(vn, 1))
                history["val_acc"].append(va / max(vn, 1))
            else:
                history["val_loss"].append(np.nan)
                history["val_acc"].append(np.nan)

            # -------- logging --------
            if epoch % print_every == 0:
                msg = f"Epoch {epoch:02d} | loss {train_loss_epoch:.4f} | acc {train_acc_epoch:.3f}"
                if val_loader is not None:
                    msg += f" | val_loss {history['val_loss'][-1]:.4f} | val_acc {history['val_acc'][-1]:.3f}"
                print(msg)

            # -------- live plot update --------
            xs = np.arange(1, len(history["train_loss"])+1)
            train_line.set_data(xs, history["train_loss"])
            if val_loader is not None:
                val_line.set_data(xs, history["val_loss"])
            ax.relim(); ax.autoscale_view()
            plt.pause(0.05)

        plt.ioff(); plt.show()
        return history


    # ------------------ inference -----------------------
    def predict(self, x, return_prob=False, threshold=0.5):
        if x.ndim < 2:
            raise ValueError("x must be (N, C_in, *spatial)")
        logits, _ = self.Forward_Propagation(x)
        if self.num_classes == 1:
            probs = self.sigmoid(logits).reshape(-1)  # (N,)
            preds = (probs >= float(threshold)).astype(np.int64)
            return (preds, probs) if return_prob else preds
        else:
            probs = self.softmax(logits)
            preds = np.argmax(probs, axis=1)
            return (preds, probs) if return_prob else preds

    # ------------------ forward -------------------------
    def Forward_Propagation(self, X, pool_every=1, conv_stride=1, conv_padding=None):
        def _build_fc(D):
            # last dim is 1 for binary, else num_classes
            last = 1 if self.num_classes == 1 else self.num_classes
            dims = [D] + self.hidden_units + [last]
            for li in range(len(dims) - 1):
                fan_in, fan_out = dims[li], dims[li+1]
                W = (np.random.randn(fan_out, fan_in) * np.sqrt(2.0/fan_in)).astype(np.float32)
                b = np.zeros((fan_out, 1), dtype=np.float32)
                self.weighted_vectors[f"W_fc{li+1}"] = W
                self.bias_vectors[f"b_fc{li+1}"]     = b
                self.gradients[f"dW_fc{li+1}"]       = np.zeros_like(W)
                self.gradients[f"db_fc{li+1}"]       = np.zeros_like(b)
            self.fc_layers = len(dims) - 1
            self.fc_built  = True
            self.fc_flatten_dim = D

        dim = self.conv_dim
        if conv_padding is None:
            conv_padding = (1,)*dim if self.use_padding else (0,)*dim
        if isinstance(conv_stride, int):
            conv_stride = (conv_stride,) * dim

        N = X.shape[0]
        feats = []
        for n in range(N):
            x = X[n]
            for i in range(1, self.conv_layers+1):
                x = self.Convolution(x, i, stride=conv_stride, padding=conv_padding)
                x = self.relu(x)
                if pool_every and (i % pool_every == 0):
                    x = self.MaxPool(x, pool_size=self.pool_size, stride=self.pool_size)
            feats.append(x)

        F = [f.reshape(-1, 1) for f in feats]
        D = F[0].shape[0]
        if (not getattr(self, "fc_built", False)) or (getattr(self, "fc_flatten_dim", D) != D):
            _build_fc(D)

        logits = []
        for n in range(N):
            z = F[n]
            for li in range(1, self.fc_layers):
                W = self.weighted_vectors[f"W_fc{li}"]
                b = self.bias_vectors[f"b_fc{li}"]
                z = self.relu(W @ z + b)
            Wl = self.weighted_vectors[f"W_fc{self.fc_layers}"]
            bl = self.bias_vectors[f"b_fc{self.fc_layers}"]
            z = Wl @ z + bl  # (last,1)
            if self.num_classes == 1:
                logits.append(z[:, 0:1])  # keep (1,1) per sample
            else:
                logits.append(z[:, 0])    # (C,)
        logits = np.stack(logits, axis=0)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)    # (N,1)
        return logits, {"flatten_dim": D, "feats": feats, "F": F}

    # ------------------ backward ------------------------
    def Backward_Propagation(self, X, y_onehot, pool_every=1, conv_stride=1, conv_padding=None):
        def _to_tuple(x, d):
            if isinstance(x, int): return (x,) * d
            x = tuple(x); assert len(x) == d, f"expected length-{d} tuple, got {x}"
            return x

        def _maxpool_backward(dout, a_before_pool, pool_size, stride):
            C = a_before_pool.shape[0]
            dim = a_before_pool.ndim - 1
            pool_size = _to_tuple(pool_size, dim)
            stride    = _to_tuple(stride, dim)
            S_out = dout.shape[1:]
            dA_prev = np.zeros_like(a_before_pool, dtype=np.float32)

            if dim == 1:
                (kW,) = pool_size
                for c in range(C):
                    for i in range(S_out[0]):
                        i0 = i * stride[0]
                        region = a_before_pool[c, i0:i0+kW]
                        idx = np.argmax(region)
                        dA_prev[c, i0 + idx] += dout[c, i]
            elif dim == 2:
                kH, kW = pool_size
                for c in range(C):
                    for i in range(S_out[0]):
                        i0 = i * stride[0]
                        for j in range(S_out[1]):
                            j0 = j * stride[1]
                            region = a_before_pool[c, i0:i0+kH, j0:j0+kW]
                            idx = np.argmax(region)
                            r, s = divmod(idx, kW)
                            dA_prev[c, i0 + r, j0 + s] += dout[c, i, j]
            else:
                kD, kH, kW = pool_size
                for c in range(C):
                    for d in range(S_out[0]):
                        d0 = d * stride[0]
                        for i in range(S_out[1]):
                            i0 = i * stride[1]
                            for j in range(S_out[2]):
                                j0 = j * stride[2]
                                region = a_before_pool[c, d0:d0+kD, i0:i0+kH, j0:j0+kW]
                                idx = np.argmax(region)
                                r1, rem = divmod(idx, kH*kW)
                                r2, r3  = divmod(rem, kW)
                                dA_prev[c, d0 + r1, i0 + r2, j0 + r3] += dout[c, d, i, j]
            return dA_prev

        def _conv_backward(dout, x, W, stride, padding):
            C_in  = x.shape[0]
            C_out = W.shape[0]
            dim = x.ndim - 1
            stride  = _to_tuple(stride, dim)
            padding = _to_tuple(padding, dim)
            S = x.shape[1:]
            K = W.shape[2:]

            pad_width = [(0,0)] + [(p,p) for p in padding]
            x_pad  = np.pad(x, pad_width, mode="constant")
            dx_pad = np.zeros_like(x_pad, dtype=np.float32)
            dW = np.zeros_like(W, dtype=np.float32)
            db = np.zeros((C_out, 1), dtype=np.float32)
            S_out = dout.shape[1:]

            if dim == 1:
                (kW,) = K
                for co in range(C_out):
                    db[co,0] = np.sum(dout[co])
                    for ci in range(C_in):
                        for i in range(S_out[0]):
                            i0 = i * stride[0]
                            region = x_pad[ci, i0:i0+kW]
                            dW[co, ci] += region * dout[co, i]
                            dx_pad[ci, i0:i0+kW] += W[co, ci] * dout[co, i]
                dx = dx_pad[(slice(None), slice(padding[0], padding[0]+S[0]))]
                return dx, dW, db

            elif dim == 2:
                kH, kW = K
                for co in range(C_out):
                    db[co,0] = np.sum(dout[co])
                    for ci in range(C_in):
                        for i in range(S_out[0]):
                            i0 = i * stride[0]
                            for j in range(S_out[1]):
                                j0 = j * stride[1]
                                region = x_pad[ci, i0:i0+kH, j0:j0+kW]
                                dW[co, ci] += region * dout[co, i, j]
                                dx_pad[ci, i0:i0+kH, j0:j0+kW] += W[co, ci] * dout[co, i, j]
                dx = dx_pad[(slice(None),
                            slice(padding[0], padding[0]+S[0]),
                            slice(padding[1], padding[1]+S[1]))]
                return dx, dW, db

            else:
                kD, kH, kW = K
                for co in range(C_out):
                    db[co,0] = np.sum(dout[co])
                    for ci in range(C_in):
                        for d in range(S_out[0]):
                            d0 = d * stride[0]
                            for i in range(S_out[1]):
                                i0 = i * stride[1]
                                for j in range(S_out[2]):
                                    j0 = j * stride[2]
                                    region = x_pad[ci, d0:d0+kD, i0:i0+kH, j0:j0+kW]
                                    dW[co, ci] += region * dout[co, d, i, j]
                                    dx_pad[ci, d0:d0+kD, i0:i0+kH, j0:j0+kW] += W[co, ci] * dout[co, d, i, j]
                dx = dx_pad[(slice(None),
                            slice(padding[0], padding[0]+S[0]),
                            slice(padding[1], padding[1]+S[1]),
                            slice(padding[2], padding[2]+S[2]))]
                return dx, dW, db

        dim = self.conv_dim
        if conv_padding is None:
            conv_padding = (1,)*dim if self.use_padding else (0,)*dim
        conv_padding = _to_tuple(conv_padding, dim)
        conv_stride  = _to_tuple(conv_stride,  dim)

        N = X.shape[0]
        conv_blocks = []
        feats = []
        for n in range(N):
            x = X[n]
            blocks = []
            for i in range(1, self.conv_layers+1):
                x_in = x
                z = self.Convolution(x_in, i, stride=conv_stride, padding=conv_padding)
                a_relu = self.relu(z)
                did_pool = False
                a_before_pool = a_relu
                if pool_every and (i % pool_every == 0):
                    a = self.MaxPool(a_relu, pool_size=self.pool_size, stride=self.pool_size)
                    did_pool = True
                else:
                    a = a_relu
                blocks.append({
                    "layer_idx": i,
                    "x_in": x_in,
                    "z": z,
                    "a": a,
                    "a_before_pool": a_before_pool,
                    "did_pool": did_pool
                })
                x = a
            feats.append(x)
            conv_blocks.append(blocks)

        F = [f.reshape(-1, 1) for f in feats]
        D = F[0].shape[0]
        if (not getattr(self, "fc_built", False)) or (getattr(self, "fc_flatten_dim", D) != D):
            last = 1 if self.num_classes == 1 else self.num_classes
            dims = [D] + self.hidden_units + [last]
            for li in range(len(dims) - 1):
                fan_in, fan_out = dims[li], dims[li+1]
                W = (np.random.randn(fan_out, fan_in) * np.sqrt(2.0/fan_in)).astype(np.float32)
                b = np.zeros((fan_out, 1), dtype=np.float32)
                self.weighted_vectors[f"W_fc{li+1}"] = W
                self.bias_vectors[f"b_fc{li+1}"]     = b
                self.gradients[f"dW_fc{li+1}"]       = np.zeros_like(W)
                self.gradients[f"db_fc{li+1}"]       = np.zeros_like(b)
            self.fc_layers = len(dims) - 1
            self.fc_built  = True
            self.fc_flatten_dim = D

        fc_caches = []
        logits = []
        for n in range(N):
            z_in = F[n]
            per = [{"z_in": z_in}]
            for li in range(1, self.fc_layers):
                W = self.weighted_vectors[f"W_fc{li}"]
                b = self.bias_vectors[f"b_fc{li}"]
                z_out = W @ z_in + b
                a = self.relu(z_out)
                per.append({"W": W, "b": b, "z_out": z_out, "a": a})
                z_in = a
            Wl = self.weighted_vectors[f"W_fc{self.fc_layers}"]
            bl = self.bias_vectors[f"b_fc{self.fc_layers}"]
            z_out = Wl @ z_in + bl
            per.append({"W": Wl, "b": bl, "z_out": z_out, "a": None})
            if self.num_classes == 1:
                logits.append(z_out[:, 0:1])  # keep (1,1)
            else:
                logits.append(z_out[:, 0])    # (C,)
            fc_caches.append(per)
        logits = np.stack(logits, axis=0)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)        # (N,1)

        # ----- loss & dLogits -----
        if self.num_classes == 1:
            probs = self.sigmoid(logits)                         # (N,1)
            loss  = self.bce_with_logits_loss(y_onehot, logits)
            dLogits = (probs - y_onehot) / N                     # (N,1)
        else:
            probs = self.softmax(logits)                         # (N,C)
            loss  = self.cross_entropy_loss(y_onehot, probs)
            dLogits = (probs - y_onehot) / N                     # (N,C)

        # zero grads
        for i in range(1, self.conv_layers+1):
            self.gradients[f"dW_conv{i}"].fill(0.0)
            self.gradients[f"db_conv{i}"].fill(0.0)
        for li in range(1, self.fc_layers+1):
            self.gradients[f"dW_fc{li}"].fill(0.0)
            self.gradients[f"db_fc{li}"].fill(0.0)

        # ----- FC backward -----
        dF_list = []
        for n in range(N):
            per = fc_caches[n]
            li = self.fc_layers
            Wl = self.weighted_vectors[f"W_fc{li}"]
            a_prev = per[-2]["a"] if li > 1 else per[0]["z_in"]
            dl = dLogits[n][:, None]                 # (C,1) or (1,1)
            self.gradients[f"dW_fc{li}"] += dl @ a_prev.T
            self.gradients[f"db_fc{li}"] += dl
            da = Wl.T @ dl

            for li2 in range(self.fc_layers-1, 0, -1):
                W = self.weighted_vectors[f"W_fc{li2}"]
                z_out = per[li2]["z_out"]
                a_in  = per[li2-1]["z_in"]
                dz = da * self.relu(z_out, derivative=True)
                self.gradients[f"dW_fc{li2}"] += dz @ a_in.T
                self.gradients[f"db_fc{li2}"] += dz
                da = W.T @ dz
            dF_list.append(da)

        # ----- Conv stack backward -----
        for n in range(N):
            grad = dF_list[n].reshape(feats[n].shape)
            for blk in reversed(conv_blocks[n]):
                i = blk["layer_idx"]
                z = blk["z"]
                a_before_pool = blk["a_before_pool"]
                if blk["did_pool"]:
                    grad = _maxpool_backward(grad, a_before_pool, pool_size=self.pool_size, stride=self.pool_size)
                grad = grad * self.relu(z, derivative=True)
                W = self.weighted_vectors[f"W_conv{i}"]
                dx, dW, db = _conv_backward(grad, blk["x_in"], W, stride=conv_stride, padding=conv_padding)
                self.gradients[f"dW_conv{i}"] += dW
                self.gradients[f"db_conv{i}"] += db
                grad = dx

        return float(loss), (self.sigmoid(logits) if self.num_classes == 1 else self.softmax(logits))

    # ------------------ conv ----------------------------
    def Convolution(self, input, conv_index, stride=1, padding=0):
        if input.ndim < 2:
            raise ValueError("Input x must be (C_in, *spatial).")
        W = self.weighted_vectors[f"W_conv{conv_index}"]
        b = self.bias_vectors[f"b_conv{conv_index}"]
        C_in = input.shape[0]
        dim  = input.ndim - 1
        if W.ndim != dim + 2:
            raise ValueError(f"W must be (C_out, C_in, *kernel_spatial) with {dim} spatial dims.")
        C_out = W.shape[0]
        if W.shape[1] != C_in:
            raise ValueError(f"W second dim (C_in={W.shape[1]}) must match x channels (C_in={C_in}).")
        if b.shape != (C_out, 1):
            raise ValueError(f"b must be shape ({C_out}, 1)")

        if isinstance(stride, int) and isinstance(padding, int):
            stride  = (stride,)  * dim
            padding = (padding,) * dim
        else:
            stride  = tuple(stride)
            padding = tuple(padding)
            if len(stride)  != dim: raise ValueError(f"Expected int or length-{dim} tuple, got {stride}")
            if len(padding) != dim: raise ValueError(f"Expected int or length-{dim} tuple, got {padding}")

        S_in    = input.shape[1:]
        K       = W.shape[2:]
        pad_w   = [(0, 0)] + [(p, p) for p in padding]
        x_pad   = np.pad(input, pad_w, mode="constant")
        S_out   = tuple((S_in[d] + 2*padding[d] - K[d]) // stride[d] + 1 for d in range(dim))
        if any(s <= 0 for s in S_out):
            raise ValueError(f"Invalid output size {S_out}. Check padding/stride/kernel vs input {S_in}.")

        y = np.zeros((C_out, *S_out), dtype=np.float32)
        for co in range(C_out):
            acc = np.zeros(S_out, dtype=np.float32)
            for ci in range(C_in):
                Kslice = W[co, ci]
                if dim == 1:
                    (kW,) = Kslice.shape
                    for i in range(S_out[0]):
                        i0 = i * stride[0]
                        region = x_pad[ci, i0:i0+kW]
                        acc[i] += np.sum(region * Kslice)
                elif dim == 2:
                    kH, kW = Kslice.shape
                    for i in range(S_out[0]):
                        i0 = i * stride[0]
                        for j in range(S_out[1]):
                            j0 = j * stride[1]
                            region = x_pad[ci, i0:i0+kH, j0:j0+kW]
                            acc[i, j] += np.sum(region * Kslice)
                else:
                    kD, kH, kW = Kslice.shape
                    for d in range(S_out[0]):
                        d0 = d * stride[0]
                        for i in range(S_out[1]):
                            i0 = i * stride[1]
                            for j in range(S_out[2]):
                                j0 = j * stride[2]
                                region = x_pad[ci, d0:d0+kD, i0:i0+kH, j0:j0+kW]
                                acc[d, i, j] += np.sum(region * Kslice)
            y[co] = acc + b[co, 0]
        return y

    # ------------------ max pool ------------------------
    def MaxPool(self, x, pool_size=2, stride=2):
        C = x.shape[0]
        dim = x.ndim - 1
        if isinstance(pool_size, int): pool_size = (pool_size,) * dim
        if isinstance(stride, int):    stride    = (stride,)    * dim
        S = x.shape[1:]
        S_out = [ (S[d] - pool_size[d]) // stride[d] + 1 for d in range(dim) ]
        y = np.zeros((C, *S_out), dtype=x.dtype)
        for c in range(C):
            for out_idx in np.ndindex(*S_out):
                slices = tuple(slice(out_idx[d]*stride[d], out_idx[d]*stride[d]+pool_size[d]) for d in range(dim))
                region = x[(c,) + slices]
                y[(c,) + out_idx] = np.max(region)
        return y

    # ------------------ utils ---------------------------
    def sigmoid(self, x):
        pos = (x >= 0)
        neg = ~pos
        z = np.empty_like(x, dtype=np.float32)
        z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        z[neg] = expx / (1.0 + expx)
        return z

    def bce_with_logits_loss(self, y, logits):
        """
        y: (N,1) with values in {0,1}
        logits: (N,1)
        Stable BCE-with-logits: softplus(logits) - y*logits
        """
        max0 = np.maximum(logits, 0)
        loss = (np.log1p(np.exp(-np.abs(logits))) + max0 - y * logits)  # shape (N,1)
        return float(np.mean(loss))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y, output):
        eps = 1e-12
        p = np.clip(output, eps, 1. - eps)
        return -np.mean(np.sum(y * np.log(p), axis=1))

    def relu(self, x, derivative=False):
        return (x > 0).astype(np.float32) if derivative else np.maximum(0, x)
