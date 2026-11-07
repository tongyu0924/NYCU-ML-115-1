"""Assignment 3, Part II: a small neural-network library written with NumPy only."""
import numpy as np

def affine_forward(X, W, b):
    """Z = XW + b.   X: (N, D), W: (D, M), b: (M,)  ->  Z: (N, M)"""
    Z = X @ W + b
    cache = (X, W)
    return Z, cache

def affine_backward(dZ, cache):
    """Returns dX (N, D), dW (D, M), db (M,)."""
    X, W = cache
    dX = dZ @ W.T
    dW = X.T @ dZ
    db = dZ.sum(axis=0)
    return dX, dW, db

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    return dA * (Z > 0)

def sigmoid_forward(Z):
    # numerically stable sigmoid
    A = np.exp(-np.logaddexp(0, -Z)).astype(Z.dtype, copy=False)
    cache = A
    return A, cache

def sigmoid_backward(dA, cache):
    A = cache
    return dA * A * (1 - A)

def softmax_ce_loss(S, y):
    """S: scores (N, C), y: integer labels (N,)."""
    N = S.shape[0]
    shifted = S - S.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(N), y].mean()
    dS = np.exp(log_probs)
    dS[np.arange(N), y] -= 1
    dS /= N
    return loss, dS

def hinge_loss(S, y, delta=1.0):
    """Multiclass hinge loss: sum_{j != y} max(0, s_j - s_y + delta), averaged."""
    N = S.shape[0]
    correct = S[np.arange(N), y][:, None]
    margins = np.maximum(0, S - correct + delta)
    margins[np.arange(N), y] = 0
    loss = margins.sum() / N
    mask = (margins > 0).astype(S.dtype)
    mask[np.arange(N), y] = -mask.sum(axis=1)
    dS = mask / N
    return loss, dS

LOSSES = {"softmax": softmax_ce_loss, "hinge": hinge_loss}
ACTIVATIONS = {
    "relu": (relu_forward, relu_backward),
    "sigmoid": (sigmoid_forward, sigmoid_backward),
}

def numerical_gradient(f, x, h=1e-5):
    """Centered finite differences of the scalar function f at x.
    x is modified in place during the computation and restored afterwards."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f(x)
        x[i] = old - h
        fm = f(x)
        x[i] = old
        grad[i] = (fp - fm) / (2 * h)
        it.iternext()
    return grad

def rel_error(a, n):
    """max_i |a_i - n_i| / max(1e-8, |a_i| + |n_i|)"""
    return np.max(np.abs(a - n) / np.maximum(1e-8, np.abs(a) + np.abs(n)))

class MLP:
    def __init__(self, input_dim, hidden_dims, num_classes=10, activation="relu",
                 init="he", std=1e-2, reg=0.0, loss="softmax",
                 dtype=np.float32, seed=0):
        """
        init: 'zero' | 'gaussian' (uses std) | 'he' (std = sqrt(2 / fan_in))
        reg : L2 strength; penalty = 0.5 * reg * sum(W**2) over all weight matrices
        """
        rng = np.random.default_rng(seed)
        self.act_f, self.act_b = ACTIVATIONS[activation]
        self.loss_fn = LOSSES[loss]
        self.reg = reg
        self.dtype = dtype
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.L = len(dims) - 1
        self.params = {}
        for l in range(1, self.L + 1):
            fan_in, fan_out = dims[l - 1], dims[l]
            if init == "zero":
                W = np.zeros((fan_in, fan_out))
            elif init == "gaussian":
                W = std * rng.standard_normal((fan_in, fan_out))
            elif init == "he":
                W = np.sqrt(2.0 / fan_in) * rng.standard_normal((fan_in, fan_out))
            else:
                raise ValueError(init)
            self.params[f"W{l}"] = W.astype(dtype)
            self.params[f"b{l}"] = np.zeros(fan_out, dtype=dtype)

    def scores(self, X):
        A = X.astype(self.dtype, copy=False)
        for l in range(1, self.L):
            A, _ = affine_forward(A, self.params[f"W{l}"], self.params[f"b{l}"])
            A, _ = self.act_f(A)
        S, _ = affine_forward(A, self.params[f"W{self.L}"], self.params[f"b{self.L}"])
        return S

    def loss(self, X, y):
        """Returns (loss, grads)."""
        A = X.astype(self.dtype, copy=False)
        caches = []
        for l in range(1, self.L):
            A, c_aff = affine_forward(A, self.params[f"W{l}"], self.params[f"b{l}"])
            A, c_act = self.act_f(A)
            caches.append((c_aff, c_act))
        S, c_last = affine_forward(A, self.params[f"W{self.L}"], self.params[f"b{self.L}"])

        data_loss, dS = self.loss_fn(S, y)
        reg_loss = 0.5 * self.reg * sum(np.sum(self.params[f"W{l}"] ** 2)
                                        for l in range(1, self.L + 1))
        grads = {}
        dA, grads[f"W{self.L}"], grads[f"b{self.L}"] = affine_backward(dS, c_last)
        for l in range(self.L - 1, 0, -1):
            c_aff, c_act = caches[l - 1]
            dZ = self.act_b(dA, c_act)
            dA, grads[f"W{l}"], grads[f"b{l}"] = affine_backward(dZ, c_aff)
        for l in range(1, self.L + 1):
            grads[f"W{l}"] = grads[f"W{l}"] + self.reg * self.params[f"W{l}"]
        return data_loss + reg_loss, grads

    def predict(self, X, batch_size=2000):
        return np.concatenate([self.scores(X[i:i + batch_size]).argmax(1)
                               for i in range(0, len(X), batch_size)])

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def eval_loss(self, X, y, batch_size=2000):
        """Data loss (no regularization) on a whole set."""
        tot = 0.0
        for i in range(0, len(X), batch_size):
            l, _ = self.loss_fn(self.scores(X[i:i + batch_size]), y[i:i + batch_size])
            tot += l * len(X[i:i + batch_size])
        return tot / len(X)

def train(model, X, y, X_val, y_val, lr=1e-3, epochs=10, batch_size=200,
          momentum=0.0, lr_decay=1.0, seed=0, verbose=True, n_train_eval=5000):
    """Plain SGD when momentum = 0.  lr is multiplied by lr_decay after each epoch.
    Returns a history dict with per-epoch train/val loss and accuracy."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    vel = {k: np.zeros_like(p) for k, p in model.params.items()}
    sub = rng.choice(N, min(n_train_eval, N), replace=False)
    hist = {"iter_loss": [], "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [], "diverged": False}
    with np.errstate(over="ignore", invalid="ignore"):
        for ep in range(epochs):
            perm = rng.permutation(N)
            ep_losses = []
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                loss, grads = model.loss(X[idx], y[idx])
                if not np.isfinite(loss):
                    hist["diverged"] = True
                    break
                for k in model.params:
                    vel[k] = momentum * vel[k] - lr * grads[k]
                    model.params[k] += vel[k]
                ep_losses.append(float(loss))
            hist["iter_loss"] += ep_losses
            if hist["diverged"]:
                if verbose:
                    print(f"epoch {ep + 1}: loss became NaN/inf -> diverged, stopping")
                break
            lr *= lr_decay
            hist["train_loss"].append(float(np.mean(ep_losses)))
            hist["val_loss"].append(model.eval_loss(X_val, y_val))
            hist["train_acc"].append(model.accuracy(X[sub], y[sub]))
            hist["val_acc"].append(model.accuracy(X_val, y_val))
            if verbose:
                print(f"epoch {ep + 1:2d}/{epochs}  train loss {hist['train_loss'][-1]:.4f}  "
                      f"train acc {hist['train_acc'][-1]:.4f}  val acc {hist['val_acc'][-1]:.4f}")
    return hist
