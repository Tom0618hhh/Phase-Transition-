# Single-GPU hybrid-network experiment with reproducibility and full experiment logging
# Study: variable hidden-layer depth, TRAINABLE-neuron ratio P, and training trajectories
# Activations: Tanh (single activation in this run)
# Dataset: Q = 12000, x in [-4, 4]
# Network width: 64 neurons per hidden layer
# Fixed and trainable parts receive the complete output of the previous layer
#
# P definition:
#   P = trainable_k / hidden_width  -> fraction of TRAINABLE neurons per layer
#   fixed_k = hidden_width - trainable_k
#
# Trainable-ratio grid (17 points):
#   0, 1/32, 2/32, 3/32, 4/32, 5/32, 6/32, 7/32, 8/32,
#   10/32, 12/32, 14/32, 16/32, 20/32, 24/32, 28/32, 32/32
#   For hidden = 64: trainable_k in
#     {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64}
#
# Initialization policy:
#   - All weight tensors and all bias vectors are drawn from ONE
#     dedicated torch.Generator seeded by experiment_seed(r, L).
#   - Draw order (two phases):
#       Phase 1: l1_w, l2_w, ..., lL_w, out_w
#       Phase 2: l1_b, l2_b, ..., lL_b, out_b
#   - Weights use Xavier-uniform with gain = 5/3.
#   - Biases use a uniform distribution with the SAME gain-scaled
#     Xavier bound as the corresponding weight tensor of that layer.
#
# P1 stability report has been removed.

import copy
import json
import math
import os
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
# 1. Experiment Configuration
# ============================================================

CONFIG = {
    "run_id_prefix": "hybrid_trainable_ratio_reproducible",
    "code_version": "v3.2.0_depth_tanh_hidden64_dense17P_gain5over3_no_P1_stability",

    "activation_list": ["tanh"],
    "n_data": 12000,
    "input_dim": 1,
    "hidden": 64,
    "layer_list": [1, 2, 3, 4, 6, 8, 10, 12, 14],

    "trainable_list": [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64],

    "repeats": 21,
    "base_seed": 42,

    "init_gain": 5.0 / 3.0,

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

    "deterministic": True,
    "precision": "float32",

    "max_workers": 16,
    "gpu_index": 0,

    "save_trajectory_data": True,
    "save_curve_plots": True,
    "save_summary_json": True,
    "save_summary_txt": True,
}


# ============================================================
# 2. Random Seed Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if CONFIG["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def experiment_seed(repeat_idx, layer_num):
    return (
        CONFIG["base_seed"]
        + repeat_idx * 100
        + layer_num
    )


# ============================================================
# 3. Dataset Construction
# ============================================================

def build_1d_dataset(n=12000):
    x = np.linspace(-4, 4, n).reshape(-1, 1)
    y = x * np.sin(x) + np.exp(-0.1 * x**2)

    return (
        x.astype(np.float32),
        y.astype(np.float32),
    )


# ============================================================
# 4. Activation Function
# ============================================================

def get_activation(name):
    name = name.lower()

    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()

    raise ValueError(f"Unsupported activation: {name}")


# ============================================================
# 5. Initialization Pool
# ============================================================

def xavier_bound(fan_in, fan_out, gain):
    return gain * math.sqrt(6.0 / (fan_in + fan_out))


def draw_weight(fan_in, fan_out, generator, gain):
    tensor = torch.empty((fan_out, fan_in), dtype=torch.float32)
    nn.init.xavier_uniform_(tensor, gain=gain, generator=generator)
    return tensor


def draw_bias(num_features, bound, generator):
    tensor = torch.empty((num_features,), dtype=torch.float32)
    tensor.uniform_(-bound, bound, generator=generator)
    return tensor


def create_shared_init_pool(hidden=64, input_dim=1, layers=4, seed=42, gain=1.0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    init_pool = {}

    init_pool["l1_w"] = draw_weight(input_dim, hidden, generator, gain)

    for i in range(2, layers + 1):
        init_pool[f"l{i}_w"] = draw_weight(hidden, hidden, generator, gain)

    init_pool["out_w"] = draw_weight(hidden, 1, generator, gain)

    b1 = xavier_bound(input_dim, hidden, gain)
    init_pool["l1_b"] = draw_bias(hidden, b1, generator)

    b_h = xavier_bound(hidden, hidden, gain)
    for i in range(2, layers + 1):
        init_pool[f"l{i}_b"] = draw_bias(hidden, b_h, generator)

    b_out = xavier_bound(hidden, 1, gain)
    init_pool["out_b"] = draw_bias(1, b_out, generator)

    return init_pool


# ============================================================
# 6. Hybrid Layer
# ============================================================

class HybridLayer(nn.Module):
    def __init__(self, in_dim, out_fixed, out_train, activation="tanh"):
        super().__init__()

        self.act = get_activation(activation)

        self.fixed_linear = (
            nn.Linear(in_dim, out_fixed) if out_fixed > 0 else None
        )

        self.train_linear = (
            nn.Linear(in_dim, out_train) if out_train > 0 else None
        )

        if self.fixed_linear is not None:
            for parameter in self.fixed_linear.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        fixed_out = (
            self.act(self.fixed_linear(x))
            if self.fixed_linear is not None
            else None
        )
        train_out = (
            self.act(self.train_linear(x))
            if self.train_linear is not None
            else None
        )
        return fixed_out, train_out


# ============================================================
# 7. Hybrid Network
# ============================================================

class HybridNet1D(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden=64,
        trainable_k=32,
        layers=4,
        activation="tanh",
    ):
        super().__init__()

        if layers < 1:
            raise ValueError("layers must be >= 1")

        if not 0 <= trainable_k <= hidden:
            raise ValueError(
                "trainable_k must satisfy 0 <= trainable_k <= hidden"
            )

        self.trainable_k = trainable_k
        self.fixed_k = hidden - trainable_k
        self.train_k = trainable_k

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
            fixed_x, train_x = layer(x)

            if fixed_x is None:
                x = train_x
            elif train_x is None:
                x = fixed_x
            else:
                x = torch.cat([fixed_x, train_x], dim=1)

        return x

    def forward(self, x):
        return self.out(self.forward_features(x))


# ============================================================
# 8. Shared Initialization
# ============================================================

def apply_shared_init_to_model(model, init_pool):
    fixed_k = model.fixed_k
    hidden = model.hidden
    layers = model.num_layers

    with torch.no_grad():
        for i in range(1, layers + 1):
            layer = model.layers[i - 1]

            full_w = init_pool[f"l{i}_w"]
            full_b = init_pool[f"l{i}_b"]

            if layer.fixed_linear is not None:
                layer.fixed_linear.weight.copy_(full_w[:fixed_k, :])
                layer.fixed_linear.bias.copy_(full_b[:fixed_k])

            if layer.train_linear is not None:
                layer.train_linear.weight.copy_(full_w[fixed_k:hidden, :])
                layer.train_linear.bias.copy_(full_b[fixed_k:hidden])

        model.out.weight.copy_(init_pool["out_w"])
        model.out.bias.copy_(init_pool["out_b"])


# ============================================================
# 9. Parameter Statistics
# ============================================================

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen = total - trainable

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
    }


# ============================================================
# 10. Hardware Information
# ============================================================

def get_hardware_info(device):
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": None,
        "gpu_memory_gb": None,
        "precision": CONFIG["precision"],
    }

    if torch.cuda.is_available():
        current_gpu_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_gpu_index)
        info["gpu_name"] = torch.cuda.get_device_name(current_gpu_index)
        info["gpu_memory_gb"] = properties.total_memory / (1024 ** 3)

    return info


# ============================================================
# 11. Single Experiment Training
# ============================================================

def train_single_experiment(args):
    (
        activation,
        layer_num,
        k,
        repeat_idx,
        init_pool,
        adam_cfg,
        task_id,
        run_id,
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

    seed = experiment_seed(repeat_idx, layer_num)
    set_seed(seed)

    task_start = time.time()

    torch.cuda.reset_peak_memory_stats()

    x, y = build_1d_dataset(CONFIG["n_data"])

    x_train = torch.from_numpy(x).to(device)
    y_train = torch.from_numpy(y).to(device)

    model = HybridNet1D(
        input_dim=CONFIG["input_dim"],
        hidden=CONFIG["hidden"],
        trainable_k=k,
        layers=layer_num,
        activation=activation,
    ).to(device)

    apply_shared_init_to_model(model, init_pool)

    parameter_stats = count_parameters(model)

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=adam_cfg["lr"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=adam_cfg["lr_factor"],
        patience=adam_cfg["lr_patience"],
        threshold=adam_cfg["lr_threshold"],
        min_lr=adam_cfg["min_lr"],
    )

    loss_fn = nn.MSELoss()

    trajectory = {
        "step": [],
        "learning_rate": [],
        "loss": [],
        "cumulative_training_time_sec": [],
    }

    best_loss = float("inf")
    best_step = 0

    epochs = adam_cfg["epochs"]

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(x_train)
        loss = loss_fn(prediction, y_train)

        loss.backward()

        if adam_cfg["gradient_clip_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=adam_cfg["gradient_clip_norm"],
            )

        optimizer.step()

        loss_value = float(loss.item())
        scheduler.step(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_step = epoch + 1

        if (
            (epoch + 1) % adam_cfg["log_every"] == 0
            or epoch == 0
            or epoch + 1 == epochs
        ):
            elapsed = time.time() - task_start
            current_lr = float(optimizer.param_groups[0]["lr"])

            trajectory["step"].append(epoch + 1)
            trajectory["learning_rate"].append(current_lr)
            trajectory["loss"].append(loss_value)
            trajectory["cumulative_training_time_sec"].append(elapsed)

        if (epoch + 1) % adam_cfg["print_every"] == 0:
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"Task {task_id:5d} | "
                f"Act={activation:4s} | "
                f"L={layer_num:2d} | "
                f"k_train={k:3d} | "
                f"P={k / CONFIG['hidden']:.4f} | "
                f"R={repeat_idx:2d} | "
                f"Step={epoch + 1:5d}/{epochs} | "
                f"LR={current_lr:.3e} | "
                f"Loss={loss_value:.6e} | "
                f"Best={best_loss:.6e}"
            )

    total_duration = time.time() - task_start

    peak_memory_allocated_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )
    peak_memory_reserved_gb = (
        torch.cuda.max_memory_reserved() / (1024 ** 3)
    )

    result = {
        "run_id": run_id,
        "task_id": int(task_id),
        "activation": activation,
        "layer": int(layer_num),
        "repeat": int(repeat_idx),
        "seed": int(seed),
        "nominal_k": int(k),
        "nominal_P": float(k / CONFIG["hidden"]),
        "actual_k": int(k),
        "actual_P": float(k / CONFIG["hidden"]),
        "fixed_neurons_per_layer": int(CONFIG["hidden"] - k),
        "hidden_width": int(CONFIG["hidden"]),
        "input_dim": int(CONFIG["input_dim"]),
        "epochs_budget": int(epochs),
        "batch_size": int(CONFIG["batch_size"]),
        "base_lr": float(adam_cfg["lr"]),
        "min_lr": float(adam_cfg["min_lr"]),
        "lr_factor": float(adam_cfg["lr_factor"]),
        "lr_patience": int(adam_cfg["lr_patience"]),
        "lr_threshold": float(adam_cfg["lr_threshold"]),
        "gradient_clip_norm": float(adam_cfg["gradient_clip_norm"]),
        "init_gain": float(CONFIG["init_gain"]),
        **parameter_stats,
        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "final_loss": float(trajectory["loss"][-1]),
        "final_lr": float(trajectory["learning_rate"][-1]),
        "epochs_used": int(epochs),
        "training_time_sec": float(total_duration),
        "peak_gpu_memory_allocated_gb": float(peak_memory_allocated_gb),
        "peak_gpu_memory_reserved_gb": float(peak_memory_reserved_gb),
        "trajectory": trajectory,
    }

    del model
    del x_train
    del y_train
    torch.cuda.empty_cache()

    return result


# ============================================================
# 12. File Utilities
# ============================================================

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_config_trajectory_data_txt(config_results, save_dir):
    first = config_results[0]
    activation = first["activation"]
    layer = first["layer"]
    k = first["actual_k"]
    P = first["actual_P"]

    curve_dir = os.path.join(
        save_dir, "training_curves", activation,
        f"layers_{layer:02d}", f"P_{P:.6f}",
    )
    os.makedirs(curve_dir, exist_ok=True)

    path = os.path.join(curve_dir, f"k_trainable_{k:03d}_trajectory_data.txt")

    sorted_results = sorted(config_results, key=lambda r: r["repeat"])

    with open(path, "w", encoding="utf-8") as file:
        file.write("=" * 140 + "\n")
        file.write(
            f"Trajectory data for Activation={activation}, "
            f"Layers={layer}, P_trainable={P:.6f}, "
            f"trainable_k={k}, fixed_k={CONFIG['hidden'] - k}\n"
        )
        file.write(f"Number of repeats: {len(sorted_results)}\n")
        file.write("=" * 140 + "\n\n")

        file.write("Raw trajectory per repeat\n")
        file.write("-" * 140 + "\n")
        file.write(
            f"{'repeat':>8} {'step':>8} {'learning_rate':>16} "
            f"{'training_loss':>20} {'cumulative_time_sec':>22}\n"
        )
        file.write("-" * 140 + "\n")

        for result in sorted_results:
            traj = result["trajectory"]
            for i in range(len(traj["step"])):
                file.write(
                    f"{result['repeat']:>8d} "
                    f"{traj['step'][i]:>8d} "
                    f"{traj['learning_rate'][i]:>16.8e} "
                    f"{traj['loss'][i]:>20.12e} "
                    f"{traj['cumulative_training_time_sec'][i]:>22.6f}\n"
                )

        file.write("\n" + "=" * 140 + "\n")
        file.write("Aggregate statistics across repeats (per logged step)\n")
        file.write("=" * 140 + "\n\n")

        steps = sorted_results[0]["trajectory"]["step"]
        losses = np.array(
            [r["trajectory"]["loss"] for r in sorted_results],
            dtype=np.float64,
        )

        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)
        min_loss = losses.min(axis=0)
        max_loss = losses.max(axis=0)

        file.write(
            f"{'step':>8} {'mean_loss':>20} {'std_loss':>20} "
            f"{'min_loss':>20} {'max_loss':>20}\n"
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

    return path


def save_config_trajectory_plot(config_results, save_dir):
    first = config_results[0]
    activation = first["activation"]
    layer = first["layer"]
    k = first["actual_k"]
    P = first["actual_P"]

    plot_dir = os.path.join(
        save_dir, "training_curve_plots", activation,
        f"layers_{layer:02d}", f"P_{P:.6f}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    sorted_results = sorted(config_results, key=lambda r: r["repeat"])

    fig, ax = plt.subplots(figsize=(12, 7))

    for result in sorted_results:
        traj = result["trajectory"]
        ax.plot(traj["step"], traj["loss"],
                linewidth=0.6, alpha=0.35, color="tab:blue")

    steps = sorted_results[0]["trajectory"]["step"]
    losses = np.array(
        [r["trajectory"]["loss"] for r in sorted_results],
        dtype=np.float64,
    )
    mean_loss = losses.mean(axis=0)

    ax.plot(steps, mean_loss, linewidth=2.5, color="tab:red",
            label="Mean across repeats")

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"Loss Curves - All {len(sorted_results)} Repeats Overlaid | "
        f"Activation={activation}, Layers={layer}, "
        f"P_trainable={P:.4f}, trainable_k={k}"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    fig.tight_layout()
    path = os.path.join(plot_dir, "loss_all_repeats.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path


def save_config_std_plot(config_results, save_dir):
    first = config_results[0]
    activation = first["activation"]
    layer = first["layer"]
    k = first["actual_k"]
    P = first["actual_P"]

    plot_dir = os.path.join(
        save_dir, "training_curve_plots", activation,
        f"layers_{layer:02d}", f"P_{P:.6f}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    sorted_results = sorted(config_results, key=lambda r: r["repeat"])

    steps = np.array(sorted_results[0]["trajectory"]["step"])
    losses = np.array(
        [r["trajectory"]["loss"] for r in sorted_results],
        dtype=np.float64,
    )
    std_loss = losses.std(axis=0)
    std_plot = np.where(std_loss <= 0, np.nan, std_loss)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, std_plot, linewidth=2.0, color="tab:orange",
            label="Loss std across repeats")

    ax.set_yscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Std of Training Loss")
    ax.set_title(
        f"Loss Std Across {len(sorted_results)} Repeats | "
        f"Activation={activation}, Layers={layer}, "
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
# 13. Summary Files
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
    write_json(path, flattened)
    return path


# ============================================================
# 14. Aggregate Statistics
# ============================================================

def aggregate_results(results):
    grouped = {}
    for result in results:
        key = (result["activation"], result["layer"], result["actual_k"])
        grouped.setdefault(key, []).append(result)

    summary = []
    for (activation, layer, k), group in sorted(grouped.items()):
        losses = np.array([it["best_loss"] for it in group], dtype=np.float64)
        durations = np.array([it["training_time_sec"] for it in group],
                             dtype=np.float64)
        summary.append({
            "activation": activation,
            "layer": layer,
            "trainable_k": k,
            "P_trainable": k / CONFIG["hidden"],
            "fixed_k": CONFIG["hidden"] - k,
            "repeats": len(group),
            "mean_best_loss": float(losses.mean()),
            "std_best_loss": float(losses.std()),
            "min_best_loss": float(losses.min()),
            "max_best_loss": float(losses.max()),
            "mean_training_time_sec": float(durations.mean()),
            "std_training_time_sec": float(durations.std()),
        })
    return summary


def save_aggregate_summary_txt(summary, save_dir):
    path = os.path.join(save_dir, "aggregate_summary.txt")
    if not summary:
        return path
    fieldnames = list(summary[0].keys())
    with open(path, "w", encoding="utf-8") as file:
        file.write("\t".join(fieldnames) + "\n")
        for row in summary:
            file.write(
                "\t".join(str(row.get(n, "")) for n in fieldnames) + "\n"
            )
    return path


# ============================================================
# 15. Plot Mean Loss vs P
# ============================================================

def plot_loss_vs_p(results, save_dir):
    for activation in CONFIG["activation_list"]:
        activation_results = [r for r in results if r["activation"] == activation]
        if not activation_results:
            continue

        grouped = {}
        for result in activation_results:
            key = (result["layer"], result["actual_k"])
            grouped.setdefault(key, []).append(result["best_loss"])

        fig, ax = plt.subplots(figsize=(12, 8))

        for layer_num in CONFIG["layer_list"]:
            p_values = []
            mean_losses = []
            for k in CONFIG["trainable_list"]:
                key = (layer_num, k)
                if key not in grouped:
                    continue
                p_values.append(k / CONFIG["hidden"])
                mean_losses.append(np.mean(grouped[key]))
            if p_values:
                ax.plot(p_values, mean_losses, "o-", linewidth=1.5,
                        markersize=4, label=f"{layer_num} Layers")

        ax.set_yscale("log")
        ax.set_xlabel("Trainable Neuron Ratio P")
        ax.set_ylabel("Mean Best Training Loss")
        ax.set_title(
            f"Mean Best Loss vs Trainable Neuron Ratio P | "
            f"Activation={activation}\n"
            f"(Q=12000, x in [-4, 4], shared initialization, "
            f"no data shuffling)"
        )
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")
        ax.set_xlim(-0.05, 1.05)

        fig.tight_layout()
        path = os.path.join(save_dir, f"mean_best_loss_vs_P_{activation}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 16. Experiment Metadata
# ============================================================

def build_metadata(run_id, save_dir, start_time, end_time, num_gpus):
    metadata = {
        "run_id": run_id,
        "code_version": CONFIG["code_version"],
        "experiment_start": start_time,
        "experiment_end": end_time,
        "output_directory": os.path.abspath(save_dir),

        "target": {
            "function": "x * sin(x) + exp(-0.1 * x^2)",
            "input_dimension": CONFIG["input_dim"],
            "data_points": CONFIG["n_data"],
            "x_range": [-4.0, 4.0],
            "sampling": "np.linspace(-4, 4, n), deterministic, no shuffle",
        },

        "network": {
            "architecture": "1-64-...-64-1",
            "hidden_width": CONFIG["hidden"],
            "hidden_layers": CONFIG["layer_list"],
            "activation_list": CONFIG["activation_list"],
            "design": (
                "Fixed and trainable neuron groups both receive the "
                "complete previous-layer output."
            ),
        },

        "P_definition": {
            "meaning": "P is the TRAINABLE neuron ratio per hidden layer.",
            "trainable_definition": "P = trainable_k / hidden_width",
            "fixed_definition": "fixed_k = hidden_width - trainable_k",
            "trainable_k_values": CONFIG["trainable_list"],
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
            "seed_formula": "base_seed + repeat_idx*100 + layer_num",
            "same_seed_across_activations": True,
            "same_init_pool_across_activations": True,
            "same_seed_across_P": True,
            "same_initialization_pool_across_P": True,
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
            "precision": CONFIG["precision"],
            "deterministic_mode": CONFIG["deterministic"],
        },
    }

    return metadata


# ============================================================
# 17. Main Experiment
# ============================================================

def run_experiment():
    print("=" * 120)
    print("Hybrid Network Experiment: Trainable-Ratio P x Depth (Tanh)")
    print("=" * 120)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This experiment requires a GPU.")

    num_gpus = torch.cuda.device_count()
    print(f"Detected GPU count: {num_gpus}")
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        print(
            f"GPU {i}: {torch.cuda.get_device_name(i)} | "
            f"Memory: {properties.total_memory / (1024 ** 3):.2f} GB"
        )

    if CONFIG["gpu_index"] >= num_gpus:
        raise RuntimeError(
            f"Requested GPU {CONFIG['gpu_index']} is not available."
        )

    hidden = CONFIG["hidden"]
    layer_list = CONFIG["layer_list"]
    trainable_list = CONFIG["trainable_list"]
    activation_list = CONFIG["activation_list"]
    repeats = CONFIG["repeats"]

    total_tasks = (
        len(activation_list) * len(layer_list)
        * len(trainable_list) * repeats
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{CONFIG['run_id_prefix']}_{timestamp}"

    save_dir = run_id
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nRun ID: {run_id}")
    print(f"Output directory: {os.path.abspath(save_dir)}")
    print(f"Activations: {activation_list}")
    print(f"Hidden layers: {layer_list}")
    print(f"Hidden width: {hidden}")
    print(f"Trainable k values: {trainable_list}")
    print(f"Number of P values: {len(trainable_list)}")
    print(f"Repeats: {repeats}")
    print(f"Total experiments: {total_tasks}")
    print(f"Training budget: {CONFIG['epochs']} steps")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Base learning rate: {CONFIG['lr']}")
    print(f"Init gain: {CONFIG['init_gain']:.6f} (= 5/3)")
    print(f"Trajectory interval: every {CONFIG['log_every']} steps")
    print("=" * 120)

    experiment_start = datetime.now().isoformat(timespec="seconds")
    global_start = time.time()

    x_np, y_np = build_1d_dataset(CONFIG["n_data"])
    np.savez_compressed(
        os.path.join(save_dir, "dataset_reference.npz"),
        x=x_np, y=y_np,
    )

    print("\nCreating shared initialization pools...")
    init_pools = {}
    for layer_num in layer_list:
        init_pools[layer_num] = []
        for repeat_idx in range(repeats):
            seed = experiment_seed(repeat_idx, layer_num)
            init_pool = create_shared_init_pool(
                hidden=hidden,
                input_dim=CONFIG["input_dim"],
                layers=layer_num,
                seed=seed,
                gain=CONFIG["init_gain"],
            )
            init_pools[layer_num].append(init_pool)
    print("Shared initialization pools created.")

    adam_cfg = {
        "epochs": CONFIG["epochs"],
        "lr": CONFIG["lr"],
        "min_lr": CONFIG["min_lr"],
        "lr_factor": CONFIG["lr_factor"],
        "lr_patience": CONFIG["lr_patience"],
        "lr_threshold": CONFIG["lr_threshold"],
        "gradient_clip_norm": CONFIG["gradient_clip_norm"],
        "print_every": CONFIG["print_every"],
        "log_every": CONFIG["log_every"],
    }

    preliminary_metadata = build_metadata(
        run_id=run_id,
        save_dir=save_dir,
        start_time=experiment_start,
        end_time=None,
        num_gpus=num_gpus,
    )
    preliminary_metadata["training"]["status"] = "running"
    write_json(
        os.path.join(save_dir, "experiment_config.json"),
        preliminary_metadata,
    )

    tasks = []
    task_id = 0
    for activation in activation_list:
        for layer_num in layer_list:
            for k in trainable_list:
                for repeat_idx in range(repeats):
                    tasks.append((
                        activation, layer_num, k, repeat_idx,
                        init_pools[layer_num][repeat_idx],
                        adam_cfg, task_id, run_id, save_dir,
                    ))
                    task_id += 1

    print(f"\nPrepared {len(tasks)} tasks.")

    results = []
    completed = 0
    last_progress_time = time.time()

    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {
            executor.submit(train_single_experiment, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            current_time = time.time()
            if (completed % 10 == 0
                    or completed == len(tasks)
                    or current_time - last_progress_time > 60):
                elapsed = current_time - global_start
                avg = elapsed / completed
                remaining = (len(tasks) - completed) * avg
                print(
                    f"\nProgress: {completed}/{len(tasks)} "
                    f"({100 * completed / len(tasks):.2f}%)"
                )
                print(
                    f"Elapsed: {elapsed / 60:.2f} min | "
                    f"Average/task: {avg:.2f} sec | "
                    f"Estimated remaining: {remaining / 60:.2f} min"
                )
                last_progress_time = current_time

    results.sort(key=lambda it: (
        it["activation"], it["layer"], it["actual_k"], it["repeat"],
    ))

    total_time = time.time() - global_start
    experiment_end = datetime.now().isoformat(timespec="seconds")

    print("\nSaving per-configuration grouped plots and data files...")
    config_groups = {}
    for result in results:
        key = (result["activation"], result["layer"], result["actual_k"])
        config_groups.setdefault(key, []).append(result)

    for key in sorted(config_groups.keys()):
        group = config_groups[key]
        if CONFIG["save_trajectory_data"]:
            save_config_trajectory_data_txt(group, save_dir)
        if CONFIG["save_curve_plots"]:
            save_config_trajectory_plot(group, save_dir)
            save_config_std_plot(group, save_dir)

    print("\nSaving summary files...")
    if CONFIG["save_summary_txt"]:
        save_summary_txt(results, save_dir)
    if CONFIG["save_summary_json"]:
        save_summary_json(results, save_dir)

    aggregate = aggregate_results(results)
    save_aggregate_summary_txt(aggregate, save_dir)

    plot_loss_vs_p(results, save_dir)

    final_metadata = build_metadata(
        run_id=run_id,
        save_dir=save_dir,
        start_time=experiment_start,
        end_time=experiment_end,
        num_gpus=num_gpus,
    )
    final_metadata["training"]["status"] = "completed"
    final_metadata["training"]["total_wall_time_sec"] = total_time
    final_metadata["training"]["total_wall_time_min"] = total_time / 60
    final_metadata["training"]["completed_tasks"] = len(results)
    final_metadata["training"]["total_tasks"] = total_tasks

    write_json(
        os.path.join(save_dir, "experiment_metadata.json"),
        final_metadata,
    )

    print("\n" + "=" * 120)
    print("Experiment completed.")
    print("=" * 120)
    print(f"Run ID: {run_id}")
    print(f"Total wall time: {total_time:.2f} sec")
    print(f"Total wall time: {total_time / 60:.2f} min")
    print(f"Completed tasks: {len(results)}/{total_tasks}")
    print(f"Results directory: {os.path.abspath(save_dir)}")
    print("=" * 120)

    return results, save_dir


# ============================================================
# 18. Entry Point
# ============================================================

def main():
    print("=" * 120)
    print("Reproducible Hybrid Network Depth Experiment (Tanh)")
    print("P is defined as the TRAINABLE neuron ratio.")
    print("=" * 120)
    results, save_dir = run_experiment()
    return results, save_dir


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    results, save_dir = main()