# ============================================================
# Fixed-depth hybrid-network experiment: width scaling
# Shared initialization across trainable ratios
# Save framework aligned with depth106.py
#
# Activation functions:
#   Tanh (single activation in this run)
#
# Hidden layers: 5
# Hidden widths: 32, 64, 128, 256
#
# Trainable ratios (P = trainable fraction), 17 points:
#   0, 1/32, 2/32, 3/32, 4/32, 5/32, 6/32, 7/32, 8/32,
#   10/32, 12/32, 14/32, 16/32, 20/32, 24/32, 28/32, 32/32
#
# Repeats: 21
#
# Target function:
#   f(x) = x*sin(x) + exp(-0.1*x^2)
#
# Domain:
#   x in [-4, 4]
#
# Training:
#   Full-batch
#   Adam
#   12000 epochs
#   No early stopping
#
# Initialization policy:
#   - All weight tensors and all bias vectors are drawn from ONE
#     dedicated torch.Generator seeded only by (repeat_idx, hidden).
#   - Draw order (two phases):
#       Phase 1: l1_w, l2_w, ..., lL_w, out_w
#       Phase 2: l1_b, l2_b, ..., lL_b, out_b
#   - Weights use Xavier-uniform with gain = 5/3.
#   - Biases use a uniform distribution with the SAME bound as the
#     corresponding weight tensor of that layer, i.e. the same
#     gain-scaled Xavier bound, so bias and weight share the same
#     initialization distribution.
#   - Because all weights are drawn first, the weight values are
#     identical to any implementation that initializes weights before
#     biases, regardless of how biases are handled.
#
# Output policy:
#   - Each (activation, hidden_width, k) configuration produces:
#       * ONE combined loss plot with all 21 repeats overlaid
#       * ONE std plot of per-step loss std across repeats
#       * ONE trajectory data txt file (raw + aggregate stats)
#   - No per-repeat curve files and no per-repeat parameter dumps.
#   - No intermediate pickle checkpoints.
#   - Aggregate plots and summary txt/json remain unchanged.
#   - All record documents are written as txt, not csv.
#
# Experimental principle:
# For every (activation, hidden_width, repeat) pair, one complete
# neuron-wise initialization pool is generated.
#
# Every P uses exactly the same:
#   1. Initial hidden-layer weight row for each neuron
#   2. Initial hidden-layer bias entry for each neuron
#   3. Output-layer initialization
#
# Therefore, changing P changes only the fixed/trainable status
# of neurons, not their initial parameter values.
#
# ============================================================

import copy
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")


# ============================================================
# 1. Global experiment configuration
# ============================================================

CONFIG = {
    "run_id_prefix": (
        "hybrid_fixed_depth5_width_scaling"
    ),

    "code_version": (
        "v8.1_shared_neuron_init_"
        "widths_32_64_128_256_depth5_"
        "tanh_trainable_ratio_"
        "grouped_repeat_plots_txt_logs_xsinx_target_"
        "dense17P_bias_xavier_gain5over3_weights_first_"
        "depth106_save_framework_no_P1_stability"
    ),

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    "target_name": (
        "x*sin(x) + exp(-0.1*x^2)"
    ),

    "target_formula": (
        "x*sin(x) + exp(-0.1*x^2)"
    ),

    "input_dim": 1,

    "output_dim": 1,

    "sampling_distribution": (
        "deterministic linspace"
    ),

    "x_range": [
        -4.0,
        4.0,
    ],

    "data_amount": 12000,

    # --------------------------------------------------------
    # Activation functions
    # --------------------------------------------------------
    "activations": [
        "tanh",
    ],

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------
    "depth": 5,

    "hidden_widths": [
        32,
        64,
        128,
        256,
    ],

    # --------------------------------------------------------
    # Trainable-ratio grid (17 points)
    #
    # P = fraction of trainable neurons per hidden layer.
    # --------------------------------------------------------
    "fixed_ratio_values": [
        0.0,       # 0/32
        0.03125,   # 1/32
        0.0625,    # 2/32
        0.09375,   # 3/32
        0.125,     # 4/32
        0.15625,   # 5/32
        0.1875,    # 6/32
        0.21875,   # 7/32
        0.25,      # 8/32
        0.3125,    # 10/32
        0.375,     # 12/32
        0.4375,    # 14/32
        0.5,       # 16/32
        0.625,     # 20/32
        0.75,      # 24/32
        0.875,     # 28/32
        1.0,       # 32/32
    ],

    "repeats": 21,

    # --------------------------------------------------------
    # Initialization gain
    #
    # Xavier-uniform with gain g uses the bound
    #     g * sqrt(6 / (fan_in + fan_out)).
    # Here g = 5/3.
    # --------------------------------------------------------
    "init_gain": 5.0 / 3.0,

    # --------------------------------------------------------
    # Randomness
    # --------------------------------------------------------
    "base_seed": 42,

    "data_seed": 12345,

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    "epochs": 12000,

    "lr": 1e-3,

    "min_lr": 1e-6,

    "lr_factor": 0.5,

    "lr_patience": 400,

    "lr_threshold": 1e-6,

    "batch_size": 12000,

    "print_every": 600,

    "log_every": 100,

    "gradient_clip_norm": 10.0,

    "early_stopping": False,

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------
    "deterministic": True,

    "precision": "float32",

    "device_type": "cuda",

    "gpu_index": 0,

    "max_workers": 16,

    "max_concurrent_gpu_tasks": 4,

    # --------------------------------------------------------
    # Save options
    # --------------------------------------------------------
    "save_trajectory_data": True,
    "save_curve_plots": True,
    "save_summary_json": True,
    "save_summary_txt": True,
}


# ============================================================
# 2. Random seed utilities
# ============================================================

def set_seed(seed):
    """
    Set Python, NumPy and PyTorch random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. Dataset
# ============================================================

def build_1d_dataset(
    n=12000,
    data_seed=12345,
):
    """
    Build deterministic 1D regression dataset.

    Target:
        f(x) = x*sin(x) + exp(-0.1*x^2)

    Domain:
        x in [-4, 4]
    """

    np.random.seed(data_seed)

    x = np.linspace(
        -4.0,
        4.0,
        n,
        dtype=np.float32,
    ).reshape(-1, 1)

    y = (
        x * np.sin(x)
        + np.exp(-0.1 * x**2)
    )

    return (
        x.astype(np.float32),
        y.astype(np.float32),
    )


# ============================================================
# 4. Activation factory
# ============================================================

def get_activation(name):
    """
    Return the requested activation module.
    """

    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    elif name == "gelu":
        return nn.GELU()

    elif name == "tanh":
        return nn.Tanh()

    raise ValueError(
        f"Unsupported activation: {name}"
    )


# ============================================================
# 5. Initialization utilities
# ============================================================

def xavier_bound(fan_in, fan_out, gain):
    """
    Gain-scaled Xavier-uniform bound:
        gain * sqrt(6 / (fan_in + fan_out))
    """

    return gain * math.sqrt(
        6.0 / (fan_in + fan_out)
    )


def draw_weight(
    fan_in,
    fan_out,
    generator,
    gain,
):
    """
    Draw a gain-scaled Xavier-uniform weight tensor of shape
    (fan_out, fan_in).
    """

    tensor = torch.empty(
        (fan_out, fan_in),
        dtype=torch.float32,
    )

    nn.init.xavier_uniform_(
        tensor,
        gain=gain,
        generator=generator,
    )

    return tensor


def draw_bias(num_features, bound, generator):
    """
    Draw a bias vector of shape (num_features,) using the SAME
    gain-scaled Xavier-uniform bound as its corresponding weight
    tensor.
    """

    tensor = torch.empty(
        (num_features,),
        dtype=torch.float32,
    )

    tensor.uniform_(
        -bound,
        bound,
        generator=generator,
    )

    return tensor


def create_uniform_init_pool(
    hidden=128,
    input_dim=1,
    layers=5,
    seed=42,
    gain=1.0,
):
    """
    Create ONE complete neuron-wise initialization pool.

    Shared across all P (and all activations) for the same
    (hidden_width, repeat) pair.

    Draw order from the dedicated generator (two phases):
        Phase 1 (all weights, in layer order):
            l1_w, l2_w, ..., lL_w, out_w
        Phase 2 (all biases, in layer order):
            l1_b, l2_b, ..., lL_b, out_b

    Weights: gain-scaled Xavier-uniform
    Biases:  Uniform with the SAME gain-scaled Xavier bound as the
             corresponding weight tensor
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    init_pool = {}

    # Phase 1: all weights
    init_pool["l1_w"] = draw_weight(
        input_dim, hidden, generator, gain,
    )

    for i in range(2, layers + 1):

        init_pool[f"l{i}_w"] = draw_weight(
            hidden, hidden, generator, gain,
        )

    init_pool["out_w"] = draw_weight(
        hidden, 1, generator, gain,
    )

    # Phase 2: all biases
    b1 = xavier_bound(input_dim, hidden, gain)
    init_pool["l1_b"] = draw_bias(
        hidden, b1, generator,
    )

    b_h = xavier_bound(hidden, hidden, gain)
    for i in range(2, layers + 1):

        init_pool[f"l{i}_b"] = draw_bias(
            hidden, b_h, generator,
        )

    b_out = xavier_bound(hidden, 1, gain)
    init_pool["out_b"] = draw_bias(
        1, b_out, generator,
    )

    return init_pool


# ============================================================
# 6. Hybrid layer
# ============================================================

class HybridLayer(nn.Module):
    """
    One hybrid hidden layer.
    """

    def __init__(
        self,
        in_dim,
        out_fixed,
        out_train,
        activation="tanh",
    ):
        super().__init__()

        self.act = get_activation(
            activation
        )

        self.fixed_linear = (
            nn.Linear(
                in_dim,
                out_fixed,
            )
            if out_fixed > 0
            else None
        )

        self.train_linear = (
            nn.Linear(
                in_dim,
                out_train,
            )
            if out_train > 0
            else None
        )

        if self.fixed_linear is not None:

            for parameter in (
                self.fixed_linear.parameters()
            ):
                parameter.requires_grad = False

    def forward(self, x):

        fixed_out = (
            self.act(
                self.fixed_linear(x)
            )
            if self.fixed_linear is not None
            else None
        )

        train_out = (
            self.act(
                self.train_linear(x)
            )
            if self.train_linear is not None
            else None
        )

        return (
            fixed_out,
            train_out,
        )


# ============================================================
# 7. Hybrid network
# ============================================================

class HybridNet1D(nn.Module):
    """
    Fixed-depth hybrid MLP.
    """

    def __init__(
        self,
        input_dim=1,
        hidden=128,
        trainable_k=64,
        layers=5,
        activation="tanh",
    ):
        super().__init__()

        if layers < 1:
            raise ValueError(
                "layers must be >= 1"
            )

        if (
            trainable_k < 0
            or trainable_k > hidden
        ):
            raise ValueError(
                "trainable_k must satisfy "
                "0 <= trainable_k <= hidden"
            )

        self.trainable_k = trainable_k
        self.fixed_k = hidden - trainable_k
        self.train_k = self.trainable_k
        self.hidden = hidden
        self.num_layers = layers
        self.activation_name = activation

        self.layers = nn.ModuleList()

        self.layers.append(
            HybridLayer(
                in_dim=input_dim,
                out_fixed=self.fixed_k,
                out_train=self.trainable_k,
                activation=activation,
            )
        )

        for _ in range(layers - 1):
            self.layers.append(
                HybridLayer(
                    in_dim=hidden,
                    out_fixed=self.fixed_k,
                    out_train=self.trainable_k,
                    activation=activation,
                )
            )

        self.out = nn.Linear(hidden, 1)

    def forward_features(self, x):

        for layer in self.layers:

            fixed_out, train_out = layer(x)

            if fixed_out is None:
                x = train_out
            elif train_out is None:
                x = fixed_out
            else:
                x = torch.cat(
                    [fixed_out, train_out],
                    dim=1,
                )

        return x

    def forward(self, x):
        return self.out(self.forward_features(x))


# ============================================================
# 8. Apply shared initialization
# ============================================================

def apply_uniform_init_to_model(
    model,
    init_pool,
):

    fixed_k = model.fixed_k
    hidden = model.hidden
    layers = model.num_layers

    with torch.no_grad():

        for i in range(1, layers + 1):

            layer = model.layers[i - 1]

            full_w = init_pool[f"l{i}_w"]
            full_b = init_pool[f"l{i}_b"]

            if layer.fixed_linear is not None:

                layer.fixed_linear.weight.copy_(
                    full_w[:fixed_k, :]
                )
                layer.fixed_linear.bias.copy_(
                    full_b[:fixed_k]
                )

            if layer.train_linear is not None:

                layer.train_linear.weight.copy_(
                    full_w[fixed_k:hidden, :]
                )
                layer.train_linear.bias.copy_(
                    full_b[fixed_k:hidden]
                )

        model.out.weight.copy_(
            init_pool["out_w"]
        )
        model.out.bias.copy_(
            init_pool["out_b"]
        )


# ============================================================
# 9. Parameter statistics
# ============================================================

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = total - trainable

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
    }


def calculate_expected_parameter_counts(
    hidden,
    k,
    layers,
    input_dim=1,
):

    train_k = k

    first_total = hidden * input_dim + hidden
    first_trainable = train_k * input_dim + train_k

    later_total = (
        (layers - 1)
        * (hidden * hidden + hidden)
    )

    later_trainable = (
        (layers - 1)
        * (train_k * hidden + train_k)
    )

    output_total = hidden + 1
    output_trainable = hidden + 1

    total = first_total + later_total + output_total
    trainable = first_trainable + later_trainable + output_trainable

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


# ============================================================
# 10. Training
# ============================================================

def train_adam(
    model,
    x_train,
    y_train,
    epochs=12000,
    lr=1e-3,
    min_lr=1e-6,
    lr_factor=0.5,
    lr_patience=400,
    lr_threshold=1e-6,
    print_every=600,
    log_every=100,
    gradient_clip_norm=10.0,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=lr_threshold,
            threshold_mode="abs",
            min_lr=min_lr,
        )
    )

    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_step = 0
    best_model_state = None
    trajectory = []

    train_start = time.time()

    for epoch in range(epochs):

        optimizer.zero_grad(set_to_none=True)

        pred = model(x_train)
        loss = loss_fn(pred, y_train)

        loss.backward()

        if gradient_clip_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()
        scheduler.step(loss)

        loss_value = float(loss.item())
        current_lr = float(optimizer.param_groups[0]["lr"])
        cumulative_time = float(time.time() - train_start)

        if loss_value < best_loss - 1e-12:

            best_loss = loss_value
            best_step = epoch + 1

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        step = epoch + 1

        if (
            step == 1
            or step % log_every == 0
            or step == epochs
        ):

            trajectory.append(
                {
                    "step": step,
                    "lr": current_lr,
                    "training_loss": loss_value,
                    "best_training_loss": float(best_loss),
                    "cumulative_training_time_sec": cumulative_time,
                }
            )

        if step % print_every == 0:

            print(
                f"    [{model.activation_name.upper()}] "
                f"Epoch {step}/{epochs}, "
                f"loss={loss_value:.6e}, "
                f"best={best_loss:.6e}, "
                f"lr={current_lr:.3e}"
            )

    if best_model_state is not None:

        model.load_state_dict(best_model_state)

    total_train_time = float(time.time() - train_start)
    final_lr = float(optimizer.param_groups[0]["lr"])

    return (
        trajectory,
        float(best_loss),
        int(best_step),
        float(final_lr),
        total_train_time,
    )


# ============================================================
# 11. GPU worker
# ============================================================

def train_single_task_single_gpu(args):

    (
        activation,
        hidden,
        k,
        repeat_idx,
        init_pool,
        adam_cfg,
        task_id,
        save_dir,
    ) = args

    gpu_index = CONFIG["gpu_index"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this worker.")

    if gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU index {gpu_index} is not available in this worker. "
            f"Visible GPU count: {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda")

    model_seed = (
        CONFIG["base_seed"]
        + repeat_idx * 1000
        + hidden * 100
    )

    set_seed(model_seed)

    data_seed = CONFIG["data_seed"]

    x, y = build_1d_dataset(
        n=CONFIG["data_amount"],
        data_seed=data_seed,
    )

    x_train = torch.from_numpy(x).to(device)
    y_train = torch.from_numpy(y).to(device)

    model = HybridNet1D(
        input_dim=CONFIG["input_dim"],
        hidden=hidden,
        trainable_k=k,
        layers=CONFIG["depth"],
        activation=activation,
    ).to(device)

    apply_uniform_init_to_model(model, init_pool)

    parameter_counts = count_parameters(model)

    expected_counts = calculate_expected_parameter_counts(
        hidden=hidden,
        k=k,
        layers=CONFIG["depth"],
        input_dim=CONFIG["input_dim"],
    )

    if parameter_counts != expected_counts:

        raise RuntimeError(
            f"Parameter-count mismatch: "
            f"activation={activation}, "
            f"hidden={hidden}, "
            f"k={k}, "
            f"actual={parameter_counts}, "
            f"expected={expected_counts}"
        )

    torch.cuda.reset_peak_memory_stats()

    (
        trajectory,
        best_loss,
        best_step,
        final_lr,
        train_time,
    ) = train_adam(
        model=model,
        x_train=x_train,
        y_train=y_train,
        epochs=adam_cfg["epochs"],
        lr=adam_cfg["lr"],
        min_lr=adam_cfg["min_lr"],
        lr_factor=adam_cfg["lr_factor"],
        lr_patience=adam_cfg["lr_patience"],
        lr_threshold=adam_cfg["lr_threshold"],
        print_every=adam_cfg["print_every"],
        log_every=adam_cfg["log_every"],
        gradient_clip_norm=adam_cfg["gradient_clip_norm"],
    )

    peak_memory_allocated = int(torch.cuda.max_memory_allocated())
    peak_memory_reserved = int(torch.cuda.max_memory_reserved())

    current_gpu_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(current_gpu_index)

    with torch.no_grad():

        final_pred = model(x_train)
        final_loss = float(
            nn.functional.mse_loss(
                final_pred, y_train
            ).item()
        )

    result = {

        "run_id": (
            f"{CONFIG['run_id_prefix']}_"
            f"{activation}_"
            f"w{hidden}_"
            f"k{k}_"
            f"r{repeat_idx:02d}"
        ),

        "code_version": CONFIG["code_version"],
        "task_id": int(task_id),
        "activation": activation,
        "hidden": int(hidden),
        "layer": int(CONFIG["depth"]),
        "depth": int(CONFIG["depth"]),
        "k": int(k),
        "P": float(k / hidden),
        "repeat_idx": int(repeat_idx),
        "seed": int(model_seed),
        "init_seed": int(CONFIG["base_seed"] + repeat_idx + hidden * 100),
        "model_seed": int(model_seed),
        "trainable_neurons_per_layer": int(k),
        "fixed_neurons_per_layer": int(hidden - k),
        "data_seed": int(data_seed),
        "init_gain": float(CONFIG["init_gain"]),

        "epochs_budget": int(adam_cfg["epochs"]),
        "batch_size": int(CONFIG["batch_size"]),
        "base_lr": float(adam_cfg["lr"]),
        "min_lr": float(adam_cfg["min_lr"]),
        "lr_factor": float(adam_cfg["lr_factor"]),
        "lr_patience": int(adam_cfg["lr_patience"]),
        "lr_threshold": float(adam_cfg["lr_threshold"]),
        "gradient_clip_norm": float(adam_cfg["gradient_clip_norm"]),

        "parameter_counts": parameter_counts,

        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "final_loss_after_best_restore": float(final_loss),
        "final_lr": float(final_lr),
        "epochs_used": int(adam_cfg["epochs"]),

        "training_time_sec": float(train_time),

        "peak_gpu_memory_allocated_bytes": peak_memory_allocated,
        "peak_gpu_memory_reserved_bytes": peak_memory_reserved,
        "peak_gpu_memory_allocated_gb": float(peak_memory_allocated / (1024 ** 3)),
        "peak_gpu_memory_reserved_gb": float(peak_memory_reserved / (1024 ** 3)),

        "gpu_index": int(current_gpu_index),
        "gpu_name": torch.cuda.get_device_name(current_gpu_index),
        "gpu_memory_gb": float(properties.total_memory / (1024 ** 3)),

        "precision": CONFIG["precision"],
        "target_formula": CONFIG["target_formula"],
        "x_range": CONFIG["x_range"],
        "data_amount": int(CONFIG["data_amount"]),

        "trajectory": trajectory,
    }

    del model
    del x_train
    del y_train
    torch.cuda.empty_cache()

    return result


# ============================================================
# 12. Per-configuration grouped plots and data
# ============================================================

def _config_dir_data(save_dir, activation, hidden, k):
    return os.path.join(
        save_dir,
        "training_curves",
        f"activation_{activation}",
        f"hidden_{hidden}",
        f"k_{k}",
    )


def _config_dir_plot(save_dir, activation, hidden, k):
    return os.path.join(
        save_dir,
        "training_curve_plots",
        f"activation_{activation}",
        f"hidden_{hidden}",
        f"k_{k}",
    )


def save_config_trajectory_data_txt(
    config_trajectories,
    activation,
    hidden,
    k,
    save_dir,
):

    P = k / hidden

    curve_dir = _config_dir_data(save_dir, activation, hidden, k)
    os.makedirs(curve_dir, exist_ok=True)

    path = os.path.join(curve_dir, "trajectory_data.txt")

    sorted_trajs = sorted(
        config_trajectories,
        key=lambda item: item["repeat_idx"],
    )

    with open(path, "w", encoding="utf-8") as file:

        file.write("=" * 140 + "\n")
        file.write(
            f"Trajectory data for Activation={activation}, "
            f"HiddenWidth={hidden}, P_trainable={P:.6f}, "
            f"trainable_k={k}, fixed_k={hidden - k}\n"
        )
        file.write(f"Number of repeats: {len(sorted_trajs)}\n")
        file.write("=" * 140 + "\n\n")

        file.write("Raw trajectory per repeat\n")
        file.write("-" * 140 + "\n")
        file.write(
            f"{'repeat':>8} {'step':>8} "
            f"{'learning_rate':>16} "
            f"{'training_loss':>20} "
            f"{'best_training_loss':>20} "
            f"{'cumulative_time_sec':>22}\n"
        )
        file.write("-" * 140 + "\n")

        for entry in sorted_trajs:
            repeat_idx = entry["repeat_idx"]
            for record in entry["trajectory"]:
                file.write(
                    f"{repeat_idx:>8d} "
                    f"{record['step']:>8d} "
                    f"{record['lr']:>16.8e} "
                    f"{record['training_loss']:>20.12e} "
                    f"{record['best_training_loss']:>20.12e} "
                    f"{record['cumulative_training_time_sec']:>22.6f}\n"
                )

        file.write("\n" + "=" * 140 + "\n")
        file.write("Aggregate statistics across repeats (per logged step)\n")
        file.write("=" * 140 + "\n\n")

        steps = np.asarray(
            [record["step"] for record in sorted_trajs[0]["trajectory"]],
            dtype=np.int64,
        )

        losses = np.asarray(
            [
                [record["training_loss"] for record in entry["trajectory"]]
                for entry in sorted_trajs
            ],
            dtype=np.float64,
        )

        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)
        min_loss = losses.min(axis=0)
        max_loss = losses.max(axis=0)

        file.write(
            f"{'step':>8} {'mean_loss':>20} "
            f"{'std_loss':>20} {'min_loss':>20} "
            f"{'max_loss':>20}\n"
        )
        file.write("-" * 140 + "\n")

        for i, step in enumerate(steps):
            file.write(
                f"{step:>8d} "
                f"{mean_loss[i]:>20.12e} "
                f"{std_loss[i]:>20.12e} "
                f"{min_loss[i]:>20.12e} "
                f"{max_loss[i]:>20.12e}\n"
            )

        file.write("\n" + "=" * 140 + "\n")
        file.write("Best-loss summary across repeats\n")
        file.write("=" * 140 + "\n\n")

        best_losses = np.asarray(
            [entry["best_loss"] for entry in sorted_trajs],
            dtype=np.float64,
        )

        file.write(f"mean_best_loss = {best_losses.mean():.12e}\n")
        file.write(f"std_best_loss  = {best_losses.std():.12e}\n")
        file.write(f"min_best_loss  = {best_losses.min():.12e}\n")
        file.write(f"max_best_loss  = {best_losses.max():.12e}\n")

    return path


def save_config_trajectory_plot(
    config_trajectories,
    activation,
    hidden,
    k,
    save_dir,
):

    P = k / hidden

    plot_dir = _config_dir_plot(save_dir, activation, hidden, k)
    os.makedirs(plot_dir, exist_ok=True)

    sorted_trajs = sorted(
        config_trajectories,
        key=lambda item: item["repeat_idx"],
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    for entry in sorted_trajs:
        traj = entry["trajectory"]
        steps = np.asarray([r["step"] for r in traj], dtype=np.int64)
        losses = np.asarray(
            [r["training_loss"] for r in traj],
            dtype=np.float64,
        )
        ax.plot(
            steps, losses,
            linewidth=0.6, alpha=0.35, color="tab:blue",
        )

    all_steps = np.asarray(
        [r["step"] for r in sorted_trajs[0]["trajectory"]],
        dtype=np.int64,
    )
    all_losses = np.asarray(
        [
            [r["training_loss"] for r in entry["trajectory"]]
            for entry in sorted_trajs
        ],
        dtype=np.float64,
    )
    mean_loss = all_losses.mean(axis=0)

    ax.plot(
        all_steps, mean_loss,
        linewidth=2.5, color="tab:red",
        label="Mean across repeats",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"Loss Curves - All {len(sorted_trajs)} Repeats Overlaid | "
        f"Activation={activation.upper()}, Hidden={hidden}, "
        f"P_trainable={P:.4f}, trainable_k={k}"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    fig.tight_layout()

    path = os.path.join(plot_dir, "loss_all_repeats.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


def save_config_std_plot(
    config_trajectories,
    activation,
    hidden,
    k,
    save_dir,
):

    P = k / hidden

    plot_dir = _config_dir_plot(save_dir, activation, hidden, k)
    os.makedirs(plot_dir, exist_ok=True)

    sorted_trajs = sorted(
        config_trajectories,
        key=lambda item: item["repeat_idx"],
    )

    steps = np.asarray(
        [r["step"] for r in sorted_trajs[0]["trajectory"]],
        dtype=np.int64,
    )

    losses = np.asarray(
        [
            [r["training_loss"] for r in entry["trajectory"]]
            for entry in sorted_trajs
        ],
        dtype=np.float64,
    )

    std_loss = losses.std(axis=0)
    std_plot = np.where(std_loss <= 0, np.nan, std_loss)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        steps, std_plot,
        linewidth=2.0, color="tab:orange",
        label="Loss std across repeats",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Std of Training Loss")
    ax.set_title(
        f"Loss Std Across {len(sorted_trajs)} Repeats | "
        f"Activation={activation.upper()}, Hidden={hidden}, "
        f"P_trainable={P:.4f}, trainable_k={k}"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    fig.tight_layout()

    path = os.path.join(plot_dir, "loss_std_all_repeats.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# 13. Aggregate summary
# ============================================================

def update_summary(all_results, result):

    activation = result["activation"]
    hidden = result["hidden"]
    k = result["k"]

    if activation not in all_results:
        all_results[activation] = {}

    if hidden not in all_results[activation]:
        all_results[activation][hidden] = {}

    if k not in all_results[activation][hidden]:
        all_results[activation][hidden][k] = {
            "P": result["P"],
            "losses": [],
            "training_times_sec": [],
            "peak_memory_allocated_bytes": [],
            "peak_memory_reserved_bytes": [],
            "repeats": [],
        }

    group = all_results[activation][hidden][k]

    group["losses"].append(result["best_loss"])
    group["training_times_sec"].append(result["training_time_sec"])
    group["peak_memory_allocated_bytes"].append(
        result["peak_gpu_memory_allocated_bytes"]
    )
    group["peak_memory_reserved_bytes"].append(
        result["peak_gpu_memory_reserved_bytes"]
    )
    group["repeats"].append(result["repeat_idx"])


def finalize_summary(all_results):

    for activation in all_results:
        for hidden in all_results[activation]:
            for k in all_results[activation][hidden]:

                group = all_results[activation][hidden][k]

                losses = np.asarray(group["losses"], dtype=np.float64)
                times = np.asarray(group["training_times_sec"], dtype=np.float64)
                allocated = np.asarray(
                    group["peak_memory_allocated_bytes"], dtype=np.float64
                )
                reserved = np.asarray(
                    group["peak_memory_reserved_bytes"], dtype=np.float64
                )

                group["mean_loss"] = float(losses.mean())
                group["std_loss"] = float(losses.std())
                group["min_loss"] = float(losses.min())
                group["max_loss"] = float(losses.max())

                group["mean_training_time_sec"] = float(times.mean())
                group["std_training_time_sec"] = float(times.std())

                group["mean_peak_gpu_memory_allocated_gb"] = float(
                    allocated.mean() / (1024 ** 3)
                )
                group["max_peak_gpu_memory_allocated_gb"] = float(
                    allocated.max() / (1024 ** 3)
                )
                group["mean_peak_gpu_memory_reserved_gb"] = float(
                    reserved.mean() / (1024 ** 3)
                )
                group["max_peak_gpu_memory_reserved_gb"] = float(
                    reserved.max() / (1024 ** 3)
                )


def save_aggregate_summary_txt(all_results, hidden_widths, fixed_lists, save_dir):

    path = os.path.join(save_dir, "aggregate_summary.txt")

    fieldnames = [
        "activation", "hidden", "trainable_k", "P_trainable", "fixed_k",
        "repeats", "mean_best_loss", "std_best_loss", "min_best_loss",
        "max_best_loss", "mean_training_time_sec", "std_training_time_sec",
        "mean_peak_gpu_memory_allocated_gb", "max_peak_gpu_memory_allocated_gb",
        "mean_peak_gpu_memory_reserved_gb", "max_peak_gpu_memory_reserved_gb",
    ]

    with open(path, "w", encoding="utf-8") as file:

        file.write("\t".join(fieldnames) + "\n")

        for activation in CONFIG["activations"]:
            for hidden in hidden_widths:
                for k in fixed_lists[hidden]:

                    data = all_results[activation][hidden][k]

                    row = [
                        activation,
                        hidden,
                        k,
                        data["P"],
                        hidden - k,
                        len(data["repeats"]),
                        data["mean_loss"],
                        data["std_loss"],
                        data["min_loss"],
                        data["max_loss"],
                        data["mean_training_time_sec"],
                        data["std_training_time_sec"],
                        data["mean_peak_gpu_memory_allocated_gb"],
                        data["max_peak_gpu_memory_allocated_gb"],
                        data["mean_peak_gpu_memory_reserved_gb"],
                        data["max_peak_gpu_memory_reserved_gb"],
                    ]

                    file.write(
                        "\t".join(str(value) for value in row) + "\n"
                    )

    return path


# ============================================================
# 14. Flatten result and save single-run summaries
# ============================================================

def flatten_result_for_summary(result):
    return {k: v for k, v in result.items() if k != "trajectory"}


def save_summary_txt(results, save_dir):
    path = os.path.join(save_dir, "experiment_summary.txt")
    rows = [flatten_result_for_summary(r) for r in results]
    if not rows:
        return path
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as file:
        file.write("\t".join(fieldnames) + "\n")
        for row in rows:
            file.write(
                "\t".join(str(row.get(n, "")) for n in fieldnames) + "\n"
            )
    return path


def save_summary_json(results, save_dir):
    path = os.path.join(save_dir, "experiment_results.json")
    flattened = [flatten_result_for_summary(r) for r in results]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(flattened, file, indent=2, ensure_ascii=False)
    return path


# ============================================================
# 15. Experiment metadata
# ============================================================

def build_metadata(run_id, save_dir, start_time, end_time, num_gpus):

    metadata = {
        "run_id": run_id,
        "code_version": CONFIG["code_version"],
        "experiment_start": start_time,
        "experiment_end": end_time,
        "output_directory": os.path.abspath(save_dir),

        "target": {
            "function": CONFIG["target_formula"],
            "input_dimension": CONFIG["input_dim"],
            "data_points": CONFIG["data_amount"],
            "x_range": CONFIG["x_range"],
            "sampling": "deterministic np.linspace, no shuffle",
        },

        "network": {
            "architecture": "1-{}-...-{}-1".format(
                CONFIG["hidden_widths"][0], CONFIG["hidden_widths"][0]
            ),
            "hidden_widths": CONFIG["hidden_widths"],
            "hidden_depth": CONFIG["depth"],
            "activation_list": CONFIG["activations"],
            "design": (
                "Fixed and trainable neuron groups both receive the "
                "complete previous-layer output."
            ),
        },

        "P_definition": {
            "meaning": "P is the TRAINABLE neuron ratio per hidden layer.",
            "trainable_definition": "P = trainable_k / hidden_width",
            "fixed_definition": "fixed_k = hidden_width - trainable_k",
            "fixed_ratio_values": CONFIG["fixed_ratio_values"],
        },

        "initialization": {
            "gain": float(CONFIG["init_gain"]),
            "draw_order": [
                "Phase 1: l1_w, l2_w, ..., lL_w, out_w",
                "Phase 2: l1_b, l2_b, ..., lL_b, out_b",
            ],
            "weights": "gain-scaled Xavier-uniform",
            "biases": (
                "uniform on the SAME gain-scaled Xavier bound as "
                "the corresponding weight tensor"
            ),
        },

        "randomness": {
            "base_seed": CONFIG["base_seed"],
            "data_seed": CONFIG["data_seed"],
            "init_seed_formula": "base_seed + repeat_idx + hidden * 100",
            "model_seed_formula": "base_seed + repeat_idx * 1000 + hidden * 100",
            "same_init_pool_across_P": True,
            "same_training_data_across_all": True,
            "same_data_order_across_all": True,
            "data_shuffle": False,
        },

        "training": {
            "optimizer": "Adam",
            "epochs_budget": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "base_learning_rate": CONFIG["lr"],
            "min_learning_rate": CONFIG["min_lr"],
            "lr_scheduler": "ReduceLROnPlateau",
            "lr_factor": CONFIG["lr_factor"],
            "lr_patience": CONFIG["lr_patience"],
            "lr_threshold": CONFIG["lr_threshold"],
            "gradient_clip_norm": CONFIG["gradient_clip_norm"],
            "trajectory_log_interval": CONFIG["log_every"],
            "print_interval": CONFIG["print_every"],
        },

        "execution": {
            "requested_gpu_index": CONFIG["gpu_index"],
            "detected_gpu_count": num_gpus,
            "max_workers": CONFIG["max_workers"],
            "max_concurrent_gpu_tasks": CONFIG["max_concurrent_gpu_tasks"],
            "precision": CONFIG["precision"],
            "deterministic_mode": CONFIG["deterministic"],
        },
    }

    return metadata


# ============================================================
# 16. Visualization (aggregate plots only)
# ============================================================

def visualize_results(
    all_results,
    hidden_widths,
    fixed_lists,
    save_dir,
    total_time,
):

    print("\nGenerating summary plots...")

    # Plot 1: Mean loss vs P
    for activation in CONFIG["activations"]:

        fig, ax = plt.subplots(figsize=(12, 8))

        for hidden in hidden_widths:

            P_values = []
            mean_losses = []
            std_losses = []

            for k in fixed_lists[hidden]:

                data = all_results[activation][hidden][k]

                P_values.append(data["P"])
                mean_losses.append(data["mean_loss"])
                std_losses.append(data["std_loss"])

            ax.errorbar(
                P_values, mean_losses, yerr=std_losses,
                fmt="o-", capsize=4, linewidth=2, markersize=5,
                label=f"Hidden width = {hidden}",
            )

        ax.set_yscale("log")
        ax.set_xlabel("Trainable Neuron Ratio P")
        ax.set_ylabel("Best Training Loss (mean)")
        ax.set_title(
            f"Best Training Loss vs Trainable Neuron Ratio\n"
            f"5 Hidden Layers, Width Scaling, "
            f"{activation.upper()}, 21 Repeats"
        )
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"loss_vs_P_{activation}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 2: Width vs loss at each P
    for activation in CONFIG["activations"]:

        fig, ax = plt.subplots(figsize=(12, 8))

        for P_index, P in enumerate(CONFIG["fixed_ratio_values"]):

            losses = []

            for hidden in hidden_widths:

                k = fixed_lists[hidden][P_index]
                losses.append(
                    all_results[activation][hidden][k]["mean_loss"]
                )

            ax.plot(
                hidden_widths, losses,
                "o-", linewidth=2, markersize=5,
                label=f"P = {P * 100:.4f}%",
            )

        ax.set_yscale("log")
        ax.set_xlabel("Hidden Width")
        ax.set_ylabel("Best Training Loss (mean)")
        ax.set_title(
            f"Best Training Loss vs Hidden Width\n"
            f"5 Hidden Layers, {activation.upper()}, 21 Repeats"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"loss_vs_width_{activation}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 3: Activation comparison at each width
    for hidden in hidden_widths:

        fig, ax = plt.subplots(figsize=(12, 8))

        for activation in CONFIG["activations"]:

            P_values = []
            mean_losses = []

            for k in fixed_lists[hidden]:

                data = all_results[activation][hidden][k]

                P_values.append(data["P"])
                mean_losses.append(data["mean_loss"])

            ax.plot(
                P_values, mean_losses,
                "o-", linewidth=2, markersize=5,
                label=activation.upper(),
            )

        ax.set_yscale("log")
        ax.set_xlabel("Trainable Neuron Ratio P")
        ax.set_ylabel("Best Training Loss (mean)")
        ax.set_title(
            f"Activation Comparison\n"
            f"5 Hidden Layers, Hidden Width = {hidden}, 21 Repeats"
        )
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"activation_comparison_width_{hidden}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # Plot 4: Relative loss vs P, normalized to P=0
    for hidden in hidden_widths:

        fig, ax = plt.subplots(figsize=(12, 8))

        for activation in CONFIG["activations"]:

            baseline_k = fixed_lists[hidden][0]
            baseline_loss = all_results[activation][hidden][baseline_k]["mean_loss"]

            P_values = []
            relative_losses = []

            for k in fixed_lists[hidden]:

                mean_loss = all_results[activation][hidden][k]["mean_loss"]

                P_values.append(
                    all_results[activation][hidden][k]["P"]
                )
                relative_losses.append(mean_loss / baseline_loss)

            ax.plot(
                P_values, relative_losses,
                "o-", linewidth=2, markersize=5,
                label=activation.upper(),
            )

        ax.axhline(1.0, linestyle="--", linewidth=1.5)

        ax.set_xlabel("Trainable Neuron Ratio P")
        ax.set_ylabel("Relative Mean Loss (Loss / Loss at P = 0)")
        ax.set_title(
            f"Relative Loss vs Trainable Ratio\n"
            f"5 Hidden Layers, Hidden Width = {hidden}"
        )
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"relative_loss_width_{hidden}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    # Human-readable summary
    summary_txt = os.path.join(save_dir, "detailed_results.txt")

    with open(summary_txt, "w", encoding="utf-8") as file:

        file.write("Fixed-depth hybrid-network width-scaling experiment\n")
        file.write("=" * 110 + "\n")

        file.write(f"Total training time: {total_time:.2f} sec\n")
        file.write(f"Hidden depth: {CONFIG['depth']}\n")
        file.write(f"Hidden widths: {hidden_widths}\n")
        file.write(f"Activations: {CONFIG['activations']}\n")
        file.write(f"Repeats: {CONFIG['repeats']}\n")
        file.write(f"Initialization gain: {CONFIG['init_gain']:.6f}\n")

        file.write("Trainable ratios: ")
        file.write(
            ", ".join(
                f"{P * 100:.4f}%" for P in CONFIG["fixed_ratio_values"]
            )
        )
        file.write("\n")

        file.write(f"Dataset: {CONFIG['target_formula']}\n")
        file.write("x range: [-4, 4]\n")
        file.write(f"Dataset size: {CONFIG['data_amount']}\n")

        file.write(
            "Initialization: same neuron-wise gain-scaled Xavier pool "
            "for all P within each (activation, width, repeat); "
            "biases use the SAME gain-scaled Xavier-uniform bound "
            "as the corresponding weights; "
            "all weights are drawn before any bias\n"
        )
        file.write(
            "Data order: deterministic np.linspace order; no shuffle\n"
        )
        file.write(
            "P meaning: fraction of trainable neurons per hidden layer\n"
        )
        file.write("=" * 110 + "\n\n")

        for activation in CONFIG["activations"]:

            file.write(f"Activation = {activation.upper()}\n")
            file.write("-" * 110 + "\n")

            for hidden in hidden_widths:

                file.write(f"\nHidden width = {hidden}\n")
                file.write("-" * 90 + "\n")

                for k in fixed_lists[hidden]:

                    data = all_results[activation][hidden][k]

                    file.write(
                        f"k={k:3d}, "
                        f"P={data['P']:.4f}, "
                        f"mean_loss={data['mean_loss']:.6e}, "
                        f"std={data['std_loss']:.6e}, "
                        f"min={data['min_loss']:.6e}, "
                        f"max={data['max_loss']:.6e}, "
                        f"mean_time={data['mean_training_time_sec']:.2f}s\n"
                    )

                file.write("\n")

            file.write("\n")

    # Machine-readable summary JSON
    json_summary = {}

    for activation in CONFIG["activations"]:

        json_summary[activation] = {}

        for hidden in hidden_widths:

            json_summary[activation][str(hidden)] = {}

            for k in fixed_lists[hidden]:

                data = all_results[activation][hidden][k]

                json_summary[activation][str(hidden)][str(k)] = {

                    "P": data["P"],
                    "mean_loss": data["mean_loss"],
                    "std_loss": data["std_loss"],
                    "min_loss": data["min_loss"],
                    "max_loss": data["max_loss"],
                    "mean_training_time_sec": data["mean_training_time_sec"],
                    "std_training_time_sec": data["std_training_time_sec"],
                    "mean_peak_gpu_memory_allocated_gb": (
                        data["mean_peak_gpu_memory_allocated_gb"]
                    ),
                    "max_peak_gpu_memory_allocated_gb": (
                        data["max_peak_gpu_memory_allocated_gb"]
                    ),
                    "mean_peak_gpu_memory_reserved_gb": (
                        data["mean_peak_gpu_memory_reserved_gb"]
                    ),
                    "max_peak_gpu_memory_reserved_gb": (
                        data["max_peak_gpu_memory_reserved_gb"]
                    ),
                    "repeats": data["repeats"],
                }

    with open(
        os.path.join(save_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(json_summary, file, indent=2, ensure_ascii=False)

    print(f"\nSummary files and plots saved to:\n{save_dir}")


# ============================================================
# 17. Main experiment
# ============================================================

def train_width_scaling_experiment():

    print("=" * 120)
    print("Fixed-depth hybrid-network width-scaling experiment")
    print("Activations: Tanh")
    print("5 hidden layers")
    print("Widths: 32 / 64 / 128 / 256")
    print("Trainable ratios: 17-point grid from 0 to 1")
    print("21 repeats")
    print("Target: x*sin(x) + exp(-0.1*x^2)")
    print("Domain: [-4, 4]")
    print(f"Init gain: {CONFIG['init_gain']:.6f} (= 5/3)")
    print("Bias initialization: gain-scaled Xavier-uniform (same bound as W)")
    print("Draw order: all weights first, then all biases")
    print("=" * 120)

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return None

    gpu_index = CONFIG["gpu_index"]

    if gpu_index >= torch.cuda.device_count():
        print(
            f"Error: requested GPU {gpu_index} is unavailable. "
            f"Visible GPU count: {torch.cuda.device_count()}"
        )
        return None

    torch.cuda.set_device(gpu_index)

    gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_memory_gb = (
        torch.cuda.get_device_properties(gpu_index).total_memory
        / (1024 ** 3)
    )

    activations = CONFIG["activations"]
    hidden_widths = CONFIG["hidden_widths"]
    P_values = CONFIG["fixed_ratio_values"]
    repeats = CONFIG["repeats"]

    print(f"GPU: {gpu_index} - {gpu_name} ({gpu_memory_gb:.1f} GB)")
    print(f"Hidden depth: {CONFIG['depth']}")
    print(f"Hidden widths: {hidden_widths}")
    print("Activations: " + ", ".join(a.upper() for a in activations))
    print(
        "Trainable ratios: "
        + ", ".join(f"{P * 100:.4f}%" for P in P_values)
    )
    print(f"Repeats: {repeats}")
    print(f"Dataset size: {CONFIG['data_amount']}")
    print("Dataset domain: [-4, 4]")
    print(f"Epochs: {CONFIG['epochs']} (no early stopping)")
    print(f"CPU worker processes: {CONFIG['max_workers']}")
    print(
        "Maximum simultaneous GPU tasks: "
        f"{CONFIG['max_concurrent_gpu_tasks']}"
    )
    print("=" * 120)

    fixed_lists = {}

    for hidden in hidden_widths:

        fixed_lists[hidden] = [
            int(round(P * hidden)) for P in P_values
        ]

        actual_P = [k / hidden for k in fixed_lists[hidden]]

        for requested, actual in zip(P_values, actual_P):
            if not np.isclose(requested, actual, atol=1e-12):
                raise RuntimeError(
                    f"P={requested} is not exactly "
                    f"representable for hidden={hidden}"
                )

        print(
            f"hidden={hidden}: trainable_k={fixed_lists[hidden]}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{CONFIG['run_id_prefix']}_{timestamp}"
    save_dir = run_id

    os.makedirs(save_dir, exist_ok=True)

    config_to_save = dict(CONFIG)

    total_tasks = (
        len(activations) * len(hidden_widths)
        * len(P_values) * repeats
    )

    config_to_save.update(
        {
            "timestamp": timestamp,
            "save_dir": save_dir,
            "trainable_neuron_lists": fixed_lists,
            "total_tasks": total_tasks,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "network_architecture": (
                "shared-input hybrid MLP; "
                "fixed and trainable neuron groups "
                "both receive the complete previous-layer output"
            ),
            "initialization_policy": (
                "one complete neuron-wise pool per "
                "(activation, hidden_width, repeat); "
                "identical across P; "
                "all weights drawn before any bias; "
                "weights use Xavier-uniform with gain = 5/3; "
                "biases use a uniform distribution with "
                "the same gain-scaled bound as the corresponding weights"
            ),
            "init_gain": float(CONFIG["init_gain"]),
            "data_order_policy": "deterministic np.linspace order; no shuffle",
            "ratio_sampling_policy": (
                "17-point dense grid from 0 to 1, "
                "1/32 spacing in the low-P region"
            ),
            "ratio_meaning": (
                "P = fraction of trainable neurons per hidden layer; "
                "k = round(P * hidden) trainable, (hidden - k) fixed"
            ),
            "output_policy": (
                "per (activation, hidden, k) configuration: "
                "one grouped loss plot, one loss-std plot, "
                "one trajectory data txt; "
                "no per-repeat files, no per-step pkl checkpoints; "
                "all record documents are txt; "
                "additional single-run summaries in experiment_results.json "
                "and experiment_summary.txt; "
                "experiment_metadata.json and dataset_reference.npz included"
            ),
        }
    )

    with open(
        os.path.join(save_dir, "experiment_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config_to_save, file, indent=2, ensure_ascii=False)

    with open(
        os.path.join(save_dir, "experiment_config.pkl"),
        "wb",
    ) as file:
        pickle.dump(config_to_save, file)

    with open(
        os.path.join(save_dir, "experiment_config.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Fixed-depth hybrid-network width-scaling experiment\n")
        file.write("=" * 100 + "\n")
        for key, value in config_to_save.items():
            file.write(f"{key}: {value}\n")

    # Save dataset reference
    x_np, y_np = build_1d_dataset(
        n=CONFIG["data_amount"],
        data_seed=CONFIG["data_seed"],
    )
    np.savez_compressed(
        os.path.join(save_dir, "dataset_reference.npz"),
        x=x_np, y=y_np,
    )

    # Pre-generate shared initialization pools
    print("\nCreating shared initialization pools...")
    init_pools = {}
    for hidden in hidden_widths:
        init_pools[hidden] = []
        for repeat_idx in range(repeats):
            init_seed = (
                CONFIG["base_seed"]
                + repeat_idx
                + hidden * 100
            )
            init_pool = create_uniform_init_pool(
                hidden=hidden,
                input_dim=CONFIG["input_dim"],
                layers=CONFIG["depth"],
                seed=init_seed,
                gain=CONFIG["init_gain"],
            )
            init_pools[hidden].append(init_pool)
    print("Shared initialization pools created.")

    adam_cfg = {
        "epochs": CONFIG["epochs"],
        "lr": CONFIG["lr"],
        "min_lr": CONFIG["min_lr"],
        "lr_factor": CONFIG["lr_factor"],
        "lr_patience": CONFIG["lr_patience"],
        "lr_threshold": CONFIG["lr_threshold"],
        "print_every": CONFIG["print_every"],
        "log_every": CONFIG["log_every"],
        "gradient_clip_norm": CONFIG["gradient_clip_norm"],
    }

    tasks = []
    task_id = 0

    for activation in activations:
        for hidden in hidden_widths:
            for k in fixed_lists[hidden]:
                for repeat_idx in range(repeats):

                    tasks.append(
                        (
                            activation,
                            hidden,
                            k,
                            repeat_idx,
                            init_pools[hidden][repeat_idx],
                            adam_cfg,
                            task_id,
                            save_dir,
                        )
                    )

                    task_id += 1

    print(f"\nTotal tasks: {len(tasks)}")
    print(
        f"Calculation: {len(activations)} activations "
        f"x {len(hidden_widths)} widths "
        f"x {len(P_values)} P values "
        f"x {repeats} repeats "
        f"= {len(tasks)} tasks"
    )

    experiment_start = datetime.now().isoformat(timespec="seconds")
    global_start_time = time.time()

    preliminary_metadata = build_metadata(
        run_id=run_id,
        save_dir=save_dir,
        start_time=experiment_start,
        end_time=None,
        num_gpus=torch.cuda.device_count(),
    )
    preliminary_metadata["training"]["status"] = "running"
    with open(
        os.path.join(save_dir, "experiment_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(preliminary_metadata, file, indent=2, ensure_ascii=False)

    all_results = {activation: {} for activation in activations}
    config_trajectories = {}
    completed = 0

    batch_size = CONFIG["max_concurrent_gpu_tasks"]
    num_batches = (len(tasks) + batch_size - 1) // batch_size

    raw_results = []

    with ProcessPoolExecutor(
        max_workers=CONFIG["max_workers"]
    ) as executor:

        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tasks))
            batch_tasks = tasks[start_idx:end_idx]

            print(
                f"\nSubmitting batch "
                f"{batch_idx + 1}/{num_batches}: "
                f"{len(batch_tasks)} tasks"
            )

            futures = {
                executor.submit(
                    train_single_task_single_gpu,
                    task,
                ): task
                for task in batch_tasks
            }

            for future in as_completed(futures):

                try:
                    result = future.result()
                except Exception as error:
                    failed_task = futures[future]
                    print("\nERROR in task:")
                    print(failed_task)
                    raise error

                update_summary(all_results, result)

                key = (
                    result["activation"],
                    result["hidden"],
                    result["k"],
                )

                config_trajectories.setdefault(key, []).append(
                    {
                        "repeat_idx": result["repeat_idx"],
                        "trajectory": result["trajectory"],
                        "best_loss": result["best_loss"],
                    }
                )

                raw_results.append(result)

                completed += 1

                elapsed = time.time() - global_start_time

                if (
                    completed == 1
                    or completed % 20 == 0
                    or completed == len(tasks)
                ):

                    avg_time = elapsed / completed
                    remaining = avg_time * (len(tasks) - completed)

                    print(
                        f"\nProgress: "
                        f"{completed}/{len(tasks)} "
                        f"({100.0 * completed / len(tasks):.2f}%)"
                    )
                    print(
                        f"Elapsed: {elapsed / 60:.1f} min | "
                        f"Estimated remaining: "
                        f"{remaining / 60:.1f} min"
                    )

    total_time = time.time() - global_start_time
    experiment_end = datetime.now().isoformat(timespec="seconds")

    finalize_summary(all_results)

    # Single-run summaries
    print("\nSaving single-run summary files...")
    if CONFIG["save_summary_txt"]:
        save_summary_txt(raw_results, save_dir)
    if CONFIG["save_summary_json"]:
        save_summary_json(raw_results, save_dir)

    # Per-config grouped plots and data files
    print("\nSaving per-configuration grouped plots and data files...")

    for key in sorted(config_trajectories.keys()):

        activation, hidden, k = key
        trajs = config_trajectories[key]

        if CONFIG["save_trajectory_data"]:
            save_config_trajectory_data_txt(
                trajs, activation, hidden, k, save_dir,
            )

        if CONFIG["save_curve_plots"]:
            save_config_trajectory_plot(
                trajs, activation, hidden, k, save_dir,
            )
            save_config_std_plot(
                trajs, activation, hidden, k, save_dir,
            )

    save_aggregate_summary_txt(
        all_results, hidden_widths, fixed_lists, save_dir,
    )

    # Final metadata
    final_metadata = build_metadata(
        run_id=run_id,
        save_dir=save_dir,
        start_time=experiment_start,
        end_time=experiment_end,
        num_gpus=torch.cuda.device_count(),
    )
    final_metadata["training"]["status"] = "completed"
    final_metadata["training"]["total_wall_time_sec"] = total_time
    final_metadata["training"]["total_wall_time_min"] = total_time / 60
    final_metadata["training"]["completed_tasks"] = len(raw_results)
    final_metadata["training"]["total_tasks"] = total_tasks

    with open(
        os.path.join(save_dir, "experiment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_metadata, file, indent=2, ensure_ascii=False)

    # Save all_results pickle
    with open(
        os.path.join(save_dir, "all_results.pkl"),
        "wb",
    ) as file:
        pickle.dump(
            {
                "all_results": all_results,
                "activations": activations,
                "hidden_widths": hidden_widths,
                "fixed_lists": fixed_lists,
                "total_time_sec": total_time,
                "completed": completed,
            },
            file,
        )

    print("\n" + "=" * 120)
    print("All training tasks completed.")
    print(f"Total time: {total_time:.2f} sec ({total_time / 60:.2f} min)")
    print(f"Average time per task: {total_time / len(tasks):.2f} sec")
    print(f"Results saved to: {save_dir}")
    print("=" * 120)

    return (
        all_results,
        raw_results,
        activations,
        hidden_widths,
        fixed_lists,
        save_dir,
        total_time,
    )


# ============================================================
# 18. Main
# ============================================================

def main():

    print("=" * 120)
    print("Hybrid-network experiment:")
    print("Tanh")
    print("5 hidden layers")
    print("Widths = 32, 64, 128, 256")
    print("Dense 17-point trainable-ratio grid")
    print("21 repeats")
    print(f"Init gain = {CONFIG['init_gain']:.6f} (= 5/3)")
    print("Bias initialization: gain-scaled Xavier-uniform (same bound as W)")
    print("Draw order: all weights first, then all biases")
    print("Target function:")
    print("x*sin(x) + exp(-0.1*x^2)")
    print("Domain: [-4, 4]")
    print("=" * 120)

    result = train_width_scaling_experiment()

    if result is None:
        print("Experiment failed.")
        return

    (
        all_results,
        raw_results,
        activations,
        hidden_widths,
        fixed_lists,
        save_dir,
        total_time,
    ) = result

    visualize_results(
        all_results=all_results,
        hidden_widths=hidden_widths,
        fixed_lists=fixed_lists,
        save_dir=save_dir,
        total_time=total_time,
    )

    print("\nExperiment completed successfully.")
    print(f"Results directory: {save_dir}")


# ============================================================
# 19. Entry point
# ============================================================

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    main()