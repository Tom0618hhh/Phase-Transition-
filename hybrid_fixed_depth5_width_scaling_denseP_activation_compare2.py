# ============================================================
# Fixed-depth hybrid-network experiment: width scaling
# Shared initialization across trainable ratios
#
# Activation functions:
#   ReLU, GeLU, Tanh
#
# Hidden layers: 5
# Hidden widths: 32, 64, 128, 256
#
# Trainable ratios (P = trainable fraction):
#   0%, 6.25%, 12.5%, 18.75%, 25%, 31.25%, 37.5%,
#   43.75%, 50%, 62.5%, 75%, 87.5%, 100%
#
# Repeats: 38
#
# Target function:
#   f(x) = sin(5x) + 0.5*cos(9x)
#        + 0.35*sin(13x - 0.7)
#        + 0.2*cos(17x + 0.4)
#        + 0.1*sin(21x)
#
# Domain:
#   x in [-2*pi, 2*pi]
#
# Training:
#   Full-batch
#   Adam
#   12000 epochs
#   No early stopping
#
# Experimental principle:
# For every (activation, hidden_width, repeat) pair, one complete
# neuron-wise Xavier initialization pool is generated.
#
# Every P uses exactly the same:
#   1. Initial hidden-layer weight row for each neuron
#   2. Initial hidden-layer bias for each neuron
#   3. Output-layer initialization
#
# Therefore, changing P changes only the fixed/trainable status
# of neurons, not their initial parameter values.
#
# ============================================================

import copy
import json
import os
import pickle
import random
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
        "v5.1_shared_neuron_init_"
        "widths_32_64_128_256_depth5_"
        "relu_gelu_tanh_new_target_trainable_ratio"
    ),

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    "target_name": (
        "sin(5x) + 0.5*cos(9x) + "
        "0.35*sin(13x-0.7) + "
        "0.2*cos(17x+0.4) + "
        "0.1*sin(21x)"
    ),

    "target_formula": (
        "sin(5x) + 0.5*cos(9x) + "
        "0.35*sin(13x - 0.7) + "
        "0.2*cos(17x + 0.4) + "
        "0.1*sin(21x)"
    ),

    "input_dim": 1,

    "output_dim": 1,

    "sampling_distribution": (
        "deterministic linspace"
    ),

    "x_range": [
        -2.0 * np.pi,
        2.0 * np.pi,
    ],

    "data_amount": 12000,

    # --------------------------------------------------------
    # Activation functions
    # --------------------------------------------------------
    "activations": [
        "relu",
        "gelu",
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
    # Trainable-ratio grid
    #
    # P = fraction of trainable neurons per hidden layer.
    # First 50% is sampled more densely.
    # All ratios are exactly representable for widths
    # 32, 64, 128 and 256.
    # --------------------------------------------------------
    "fixed_ratio_values": [
        0.0,
        0.0625,
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
    ],

    "repeats": 38,

    # --------------------------------------------------------
    # Randomness
    # --------------------------------------------------------
    "base_seed": 42,

    "data_seed": 12345,

    # Important:
    # k/P is NOT included in the initialization seed.
    #
    # The initialization is shared across P.
    "init_seed_formula": (
        "base_seed + repeat_idx + hidden_width*100"
    ),

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
    # Storage
    # --------------------------------------------------------
    "save_full_loss_curve": True,

    "save_trajectory": True,
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
    Build a deterministic 1D regression dataset.

    Target:
        f(x) = sin(5x)
             + 0.5*cos(9x)
             + 0.35*sin(13x - 0.7)
             + 0.2*cos(17x + 0.4)
             + 0.1*sin(21x)

    Domain:
        x in [-2*pi, 2*pi]
    """

    # The target is deterministic.
    # data_seed is still recorded for experiment identity.
    np.random.seed(data_seed)

    x = np.linspace(
        -2.0 * np.pi,
        2.0 * np.pi,
        n,
        dtype=np.float32,
    ).reshape(-1, 1)

    y = (
        np.sin(5.0 * x)
        + 0.5 * np.cos(9.0 * x)
        + 0.35 * np.sin(
            13.0 * x - 0.7
        )
        + 0.2 * np.cos(
            17.0 * x + 0.4
        )
        + 0.1 * np.sin(
            21.0 * x
        )
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

def xavier_tensor(
    shape,
    generator,
):
    """
    Create a Xavier-uniform tensor using
    a dedicated CPU generator.
    """

    tensor = torch.empty(
        shape,
        dtype=torch.float32,
    )

    nn.init.xavier_uniform_(
        tensor,
        generator=generator,
    )

    return tensor


def zeros_tensor(shape):
    """
    Create zero initialization.
    """

    return torch.zeros(
        shape,
        dtype=torch.float32,
    )


def create_uniform_init_pool(
    hidden=128,
    input_dim=1,
    layers=5,
    seed=42,
):
    """
    Create ONE complete neuron-wise initialization pool.

    For every hidden layer:

        weight shape = [hidden, input_dim]

    for layer 1, and

        weight shape = [hidden, hidden]

    for later layers.

    Neuron j always receives the same initial row,
    regardless of trainable ratio P.

    The output layer is also initialized once and
    shared across all P values.

    This guarantees:

        Same width
        Same repeat
        Same initialization
        Different P only changes parameter freezing
    """

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(seed)

    init_pool = {}

    # --------------------------------------------------------
    # First hidden layer
    # --------------------------------------------------------

    init_pool["l1_w"] = xavier_tensor(
        (
            hidden,
            input_dim,
        ),
        generator,
    )

    init_pool["l1_b"] = zeros_tensor(
        (hidden,)
    )

    # --------------------------------------------------------
    # Hidden layers 2..depth
    # --------------------------------------------------------

    for i in range(
        2,
        layers + 1,
    ):
        init_pool[f"l{i}_w"] = xavier_tensor(
            (
                hidden,
                hidden,
            ),
            generator,
        )

        init_pool[f"l{i}_b"] = zeros_tensor(
            (hidden,)
        )

    # --------------------------------------------------------
    # Output layer
    # --------------------------------------------------------

    init_pool["out_w"] = xavier_tensor(
        (
            1,
            hidden,
        ),
        generator,
    )

    init_pool["out_b"] = zeros_tensor(
        (1,)
    )

    return init_pool


# ============================================================
# 6. Hybrid layer
# ============================================================

class HybridLayer(nn.Module):
    """
    One hybrid hidden layer.

    The output neurons are split into:

        fixed neurons
        trainable neurons

    Both groups receive the complete previous-layer output.
    """

    def __init__(
        self,
        in_dim,
        out_fixed,
        out_train,
        activation="gelu",
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

        # Fixed neurons do not receive gradients.
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

    Every hidden layer has exactly 'hidden' neurons.

    The first (hidden - trainable_k) neurons are fixed.
    The remaining trainable_k neurons are trainable.

    Both groups receive the complete output of
    the previous layer.

    P = trainable_k / hidden is the trainable neuron ratio.
    """

    def __init__(
        self,
        input_dim=1,
        hidden=128,
        trainable_k=64,
        layers=5,
        activation="gelu",
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

        # trainable_k is now the number of TRAINABLE neurons.
        self.trainable_k = trainable_k

        # fixed_k is the number of FIXED neurons.
        self.fixed_k = (
            hidden - trainable_k
        )

        # Alias for backward compatibility.
        self.train_k = self.trainable_k

        self.hidden = hidden

        self.num_layers = layers

        self.activation_name = activation

        self.layers = nn.ModuleList()

        # ----------------------------------------------------
        # Hidden layer 1
        # ----------------------------------------------------

        self.layers.append(
            HybridLayer(
                in_dim=input_dim,
                out_fixed=self.fixed_k,
                out_train=self.trainable_k,
                activation=activation,
            )
        )

        # ----------------------------------------------------
        # Hidden layers 2..depth
        # ----------------------------------------------------

        for _ in range(
            layers - 1
        ):
            self.layers.append(
                HybridLayer(
                    in_dim=hidden,
                    out_fixed=self.fixed_k,
                    out_train=self.trainable_k,
                    activation=activation,
                )
            )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.out = nn.Linear(
            hidden,
            1,
        )

    def forward_features(self, x):

        for layer in self.layers:

            fixed_out, train_out = (
                layer(x)
            )

            if fixed_out is None:

                x = train_out

            elif train_out is None:

                x = fixed_out

            else:

                x = torch.cat(
                    [
                        fixed_out,
                        train_out,
                    ],
                    dim=1,
                )

        return x

    def forward(self, x):

        return self.out(
            self.forward_features(x)
        )


# ============================================================
# 8. Apply shared initialization
# ============================================================

def apply_uniform_init_to_model(
    model,
    init_pool,
):
    """
    Assign the shared neuron-wise initialization pool.

    Neuron indices:

        [0 : hidden-trainable_k)        -> fixed
        [hidden-trainable_k : hidden)   -> trainable

    Changing trainable_k does NOT regenerate the initial
    weights; it only re-partitions the same pool.
    """

    fixed_k = model.fixed_k

    hidden = model.hidden

    layers = model.num_layers

    with torch.no_grad():

        # ----------------------------------------------------
        # Hidden layers
        # ----------------------------------------------------

        for i in range(
            1,
            layers + 1,
        ):

            layer = model.layers[
                i - 1
            ]

            full_w = init_pool[
                f"l{i}_w"
            ]

            full_b = init_pool[
                f"l{i}_b"
            ]

            # Fixed neurons occupy the first fixed_k rows
            if layer.fixed_linear is not None:

                layer.fixed_linear.weight.copy_(
                    full_w[:fixed_k, :]
                )

                layer.fixed_linear.bias.copy_(
                    full_b[:fixed_k]
                )

            # Trainable neurons occupy the remaining rows
            if layer.train_linear is not None:

                layer.train_linear.weight.copy_(
                    full_w[fixed_k:hidden, :]
                )

                layer.train_linear.bias.copy_(
                    full_b[fixed_k:hidden]
                )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

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
        "total_parameters": int(
            total
        ),

        "trainable_parameters": int(
            trainable
        ),

        "frozen_parameters": int(
            frozen
        ),
    }


def calculate_expected_parameter_counts(
    hidden,
    k,
    layers,
    input_dim=1,
):
    """
    Calculate theoretical parameter counts.

    Here k is the number of TRAINABLE neurons
    per hidden layer.
    """

    train_k = k

    # First hidden layer
    first_total = (
        hidden * input_dim
        + hidden
    )

    first_trainable = (
        train_k * input_dim
        + train_k
    )

    # Remaining hidden layers
    later_total = (
        (layers - 1)
        * (
            hidden * hidden
            + hidden
        )
    )

    later_trainable = (
        (layers - 1)
        * (
            train_k * hidden
            + train_k
        )
    )

    # Output layer
    output_total = (
        hidden + 1
    )

    output_trainable = (
        hidden + 1
    )

    total = (
        first_total
        + later_total
        + output_total
    )

    trainable = (
        first_trainable
        + later_trainable
        + output_trainable
    )

    return {
        "total_parameters": int(
            total
        ),

        "trainable_parameters": int(
            trainable
        ),

        "frozen_parameters": int(
            total - trainable
        ),
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
    """
    Full-batch Adam training.

    There is NO early stopping.

    All epochs are executed.

    The best state is restored only after
    the complete training budget finishes.
    """

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

    best_model_state = None

    full_loss_history = []

    trajectory = []

    train_start = time.time()

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            x_train
        )

        loss = loss_fn(
            pred,
            y_train,
        )

        loss.backward()

        if gradient_clip_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        optimizer.step()

        scheduler.step(loss)

        loss_value = float(
            loss.item()
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        cumulative_time = float(
            time.time()
            - train_start
        )

        full_loss_history.append(
            loss_value
        )

        # ----------------------------------------------------
        # Track best model
        # ----------------------------------------------------

        if (
            loss_value
            < best_loss - 1e-12
        ):

            best_loss = loss_value

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

        step = epoch + 1

        # ----------------------------------------------------
        # Compact trajectory
        # ----------------------------------------------------

        if (
            step == 1
            or step % log_every == 0
            or step == epochs
        ):

            trajectory.append(
                {
                    "step": step,

                    "lr": current_lr,

                    "training_loss": (
                        loss_value
                    ),

                    "best_training_loss": (
                        float(best_loss)
                    ),

                    "cumulative_training_time_sec": (
                        cumulative_time
                    ),
                }
            )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        if (
            step % print_every
            == 0
        ):

            print(
                f"    [{model.activation_name.upper()}] "
                f"Epoch {step}/{epochs}, "
                f"loss={loss_value:.6e}, "
                f"best={best_loss:.6e}, "
                f"lr={current_lr:.3e}"
            )

    # --------------------------------------------------------
    # No early stopping.
    # Restore best state only after all epochs.
    # --------------------------------------------------------

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

    total_train_time = float(
        time.time()
        - train_start
    )

    return (
        np.asarray(
            full_loss_history,
            dtype=np.float32,
        ),

        trajectory,

        float(best_loss),

        total_train_time,
    )


# ============================================================
# 11. GPU worker
# ============================================================

def train_single_task_single_gpu(
    args
):
    """
    Train one:

        activation
        width
        k (trainable neurons per hidden layer)
        repeat

    combination.
    """

    (
        activation,
        hidden,
        k,
        repeat_idx,
        adam_cfg,
        task_id,
        save_dir,
    ) = args

    gpu_index = CONFIG[
        "gpu_index"
    ]

    # --------------------------------------------------------
    # CUDA checks
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available "
            "in this worker."
        )

    if (
        gpu_index
        >= torch.cuda.device_count()
    ):

        raise RuntimeError(
            f"Requested GPU index "
            f"{gpu_index} is not available "
            f"in this worker. "
            f"Visible GPU count: "
            f"{torch.cuda.device_count()}"
        )

    torch.cuda.set_device(
        gpu_index
    )

    device = torch.device(
        "cuda"
    )

    # --------------------------------------------------------
    # Model seed
    #
    # IMPORTANT:
    # k/P is NOT included.
    # --------------------------------------------------------

    model_seed = (
        CONFIG["base_seed"]
        + repeat_idx * 1000
        + hidden * 100
    )

    set_seed(
        model_seed
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    data_seed = CONFIG[
        "data_seed"
    ]

    x, y = build_1d_dataset(
        n=CONFIG[
            "data_amount"
        ],

        data_seed=data_seed,
    )

    x_train = (
        torch.from_numpy(x)
        .to(device)
    )

    y_train = (
        torch.from_numpy(y)
        .to(device)
    )

    # --------------------------------------------------------
    # Shared initialization pool
    #
    # k is deliberately absent from the seed.
    # --------------------------------------------------------

    init_seed = (
        CONFIG["base_seed"]
        + repeat_idx
        + hidden * 100
    )

    init_pool = (
        create_uniform_init_pool(
            hidden=hidden,
            input_dim=CONFIG[
                "input_dim"
            ],
            layers=CONFIG[
                "depth"
            ],
            seed=init_seed,
        )
    )

    # --------------------------------------------------------
    # Model
    #
    # k is the number of TRAINABLE neurons per hidden layer.
    # --------------------------------------------------------

    model = HybridNet1D(
        input_dim=CONFIG[
            "input_dim"
        ],

        hidden=hidden,

        trainable_k=k,

        layers=CONFIG[
            "depth"
        ],

        activation=activation,
    ).to(device)

    # --------------------------------------------------------
    # Apply exactly the same initialization
    # for all P values.
    # --------------------------------------------------------

    apply_uniform_init_to_model(
        model,
        init_pool,
    )

    # --------------------------------------------------------
    # Save exact initial parameters
    # --------------------------------------------------------

    initial_parameter_snapshot = {
        name: parameter.detach()
        .cpu()
        .numpy()
        .copy()

        for name, parameter
        in model.named_parameters()
    }

    # --------------------------------------------------------
    # Parameter count verification
    # --------------------------------------------------------

    parameter_counts = (
        count_parameters(
            model
        )
    )

    expected_counts = (
        calculate_expected_parameter_counts(
            hidden=hidden,
            k=k,
            layers=CONFIG[
                "depth"
            ],
            input_dim=CONFIG[
                "input_dim"
            ],
        )
    )

    if (
        parameter_counts
        != expected_counts
    ):

        raise RuntimeError(
            f"Parameter-count mismatch: "
            f"activation={activation}, "
            f"hidden={hidden}, "
            f"k={k}, "
            f"actual={parameter_counts}, "
            f"expected={expected_counts}"
        )

    # --------------------------------------------------------
    # GPU memory tracking
    # --------------------------------------------------------

    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        loss_history,
        trajectory,
        best_loss,
        train_time,
    ) = train_adam(
        model=model,

        x_train=x_train,

        y_train=y_train,

        epochs=adam_cfg[
            "epochs"
        ],

        lr=adam_cfg[
            "lr"
        ],

        min_lr=adam_cfg[
            "min_lr"
        ],

        lr_factor=adam_cfg[
            "lr_factor"
        ],

        lr_patience=adam_cfg[
            "lr_patience"
        ],

        lr_threshold=adam_cfg[
            "lr_threshold"
        ],

        print_every=adam_cfg[
            "print_every"
        ],

        log_every=adam_cfg[
            "log_every"
        ],

        gradient_clip_norm=adam_cfg[
            "gradient_clip_norm"
        ],
    )

    # --------------------------------------------------------
    # GPU memory
    # --------------------------------------------------------

    peak_memory_allocated = int(
        torch.cuda.max_memory_allocated()
    )

    peak_memory_reserved = int(
        torch.cuda.max_memory_reserved()
    )

    current_gpu_index = (
        torch.cuda.current_device()
    )

    properties = (
        torch.cuda.get_device_properties(
            current_gpu_index
        )
    )

    # --------------------------------------------------------
    # Final / best-restored parameters
    # --------------------------------------------------------

    final_parameter_snapshot = {
        name: parameter.detach()
        .cpu()
        .numpy()
        .copy()

        for name, parameter
        in model.named_parameters()
    }

    # --------------------------------------------------------
    # Final loss after restoring best state
    # --------------------------------------------------------

    with torch.no_grad():

        final_pred = model(
            x_train
        )

        final_loss = float(
            nn.functional.mse_loss(
                final_pred,
                y_train,
            ).item()
        )

    # --------------------------------------------------------
    # Build result dictionary
    # --------------------------------------------------------

    result = {

        # ====================================================
        # Experiment identity
        # ====================================================

        "run_id": (
            f"{CONFIG['run_id_prefix']}_"
            f"{activation}_"
            f"w{hidden}_"
            f"k{k}_"
            f"r{repeat_idx:02d}"
        ),

        "code_version": (
            CONFIG["code_version"]
        ),

        "task_id": int(
            task_id
        ),

        # ====================================================
        # Architecture
        # ====================================================

        "activation": activation,

        "hidden": int(
            hidden
        ),

        "depth": int(
            CONFIG["depth"]
        ),

        # k now means trainable neurons per hidden layer.
        "k": int(
            k
        ),

        # P now means trainable neuron ratio.
        "P": float(
            k / hidden
        ),

        "repeat_idx": int(
            repeat_idx
        ),

        "trainable_neurons_per_layer": int(
            k
        ),

        "fixed_neurons_per_layer": int(
            hidden - k
        ),

        # ====================================================
        # Randomness
        # ====================================================

        "data_seed": int(
            data_seed
        ),

        "init_seed": int(
            init_seed
        ),

        "model_seed": int(
            model_seed
        ),

        # ====================================================
        # Parameter counts
        # ====================================================

        "parameter_counts": (
            parameter_counts
        ),

        # ====================================================
        # Training results
        # ====================================================

        "best_loss": float(
            best_loss
        ),

        "final_loss_after_best_restore": (
            float(final_loss)
        ),

        "epochs": int(
            len(loss_history)
        ),

        "training_time_sec": float(
            train_time
        ),

        # ====================================================
        # Hardware
        # ====================================================

        "peak_gpu_memory_allocated_bytes": (
            peak_memory_allocated
        ),

        "peak_gpu_memory_reserved_bytes": (
            peak_memory_reserved
        ),

        "gpu_index": int(
            current_gpu_index
        ),

        "gpu_name": (
            torch.cuda.get_device_name(
                current_gpu_index
            )
        ),

        "gpu_memory_gb": float(
            properties.total_memory
            / (1024 ** 3)
        ),

        "precision": (
            CONFIG["precision"]
        ),

        # ====================================================
        # Dataset information
        # ====================================================

        "target_formula": (
            CONFIG["target_formula"]
        ),

        "x_range": (
            CONFIG["x_range"]
        ),

        "data_amount": int(
            CONFIG["data_amount"]
        ),

        # ====================================================
        # Training curves
        # ====================================================

        "loss_history": (
            loss_history
        ),

        "trajectory": (
            trajectory
        ),

        # ====================================================
        # Parameter distributions
        # ====================================================

        "initial_parameter_snapshot": (
            initial_parameter_snapshot
        ),

        "final_parameter_snapshot": (
            final_parameter_snapshot
        ),
    }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model

    del x_train

    del y_train

    torch.cuda.empty_cache()

    return result


# ============================================================
# 12. Save one completed run
# ============================================================

def save_run_result(
    result,
    save_dir,
):
    """
    Save:

        1. Full loss curve
        2. Compact trajectory
        3. Initial parameters
        4. Final parameters
        5. Run metadata
    """

    activation = result[
        "activation"
    ]

    hidden = result[
        "hidden"
    ]

    k = result[
        "k"
    ]

    repeat_idx = result[
        "repeat_idx"
    ]

    run_dir = os.path.join(
        save_dir,
        "runs",
        f"activation_{activation}",
        f"hidden_{hidden}",
        f"k_{k}",
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Loss curve
    # --------------------------------------------------------

    curve_path = os.path.join(
        run_dir,
        f"repeat_{repeat_idx:02d}.npz",
    )

    trajectory = result[
        "trajectory"
    ]

    trajectory_steps = np.asarray(
        [
            item["step"]
            for item in trajectory
        ],
        dtype=np.int32,
    )

    trajectory_lr = np.asarray(
        [
            item["lr"]
            for item in trajectory
        ],
        dtype=np.float64,
    )

    trajectory_loss = np.asarray(
        [
            item["training_loss"]
            for item in trajectory
        ],
        dtype=np.float32,
    )

    trajectory_best_loss = np.asarray(
        [
            item["best_training_loss"]
            for item in trajectory
        ],
        dtype=np.float32,
    )

    trajectory_time = np.asarray(
        [
            item[
                "cumulative_training_time_sec"
            ]
            for item in trajectory
        ],
        dtype=np.float64,
    )

    np.savez_compressed(
        curve_path,

        loss_history=result[
            "loss_history"
        ],

        trajectory_step=(
            trajectory_steps
        ),

        trajectory_lr=(
            trajectory_lr
        ),

        trajectory_training_loss=(
            trajectory_loss
        ),

        trajectory_best_training_loss=(
            trajectory_best_loss
        ),

        trajectory_cumulative_training_time_sec=(
            trajectory_time
        ),
    )

    # --------------------------------------------------------
    # Parameter snapshots
    # --------------------------------------------------------

    parameter_path = os.path.join(
        run_dir,
        f"repeat_{repeat_idx:02d}_parameters.npz",
    )

    parameter_arrays = {}

    for (
        name,
        array
    ) in result[
        "initial_parameter_snapshot"
    ].items():

        safe_name = (
            "initial__"
            + name.replace(
                ".",
                "__",
            )
        )

        parameter_arrays[
            safe_name
        ] = array

    for (
        name,
        array
    ) in result[
        "final_parameter_snapshot"
    ].items():

        safe_name = (
            "final__"
            + name.replace(
                ".",
                "__",
            )
        )

        parameter_arrays[
            safe_name
        ] = array

    np.savez_compressed(
        parameter_path,
        **parameter_arrays,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        key: value

        for key, value
        in result.items()

        if key not in {
            "loss_history",
            "trajectory",
            "initial_parameter_snapshot",
            "final_parameter_snapshot",
        }
    }

    metadata[
        "curve_file"
    ] = os.path.relpath(
        curve_path,
        save_dir,
    )

    metadata[
        "parameter_file"
    ] = os.path.relpath(
        parameter_path,
        save_dir,
    )

    metadata_path = os.path.join(
        run_dir,
        f"repeat_{repeat_idx:02d}_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 13. Aggregate summary
# ============================================================

def update_summary(
    all_results,
    result,
):
    """
    Add one completed run to the aggregate summary.
    """

    activation = result[
        "activation"
    ]

    hidden = result[
        "hidden"
    ]

    k = result[
        "k"
    ]

    if activation not in all_results:

        all_results[
            activation
        ] = {}

    if hidden not in all_results[
        activation
    ]:

        all_results[
            activation
        ][hidden] = {}

    if k not in all_results[
        activation
    ][hidden]:

        all_results[
            activation
        ][hidden][k] = {

            "P": result[
                "P"
            ],

            "losses": [],

            "training_times_sec": [],

            "peak_memory_allocated_bytes": [],

            "peak_memory_reserved_bytes": [],

            "repeats": [],
        }

    group = all_results[
        activation
    ][hidden][k]

    group[
        "losses"
    ].append(
        result[
            "best_loss"
        ]
    )

    group[
        "training_times_sec"
    ].append(
        result[
            "training_time_sec"
        ]
    )

    group[
        "peak_memory_allocated_bytes"
    ].append(
        result[
            "peak_gpu_memory_allocated_bytes"
        ]
    )

    group[
        "peak_memory_reserved_bytes"
    ].append(
        result[
            "peak_gpu_memory_reserved_bytes"
        ]
    )

    group[
        "repeats"
    ].append(
        result[
            "repeat_idx"
        ]
    )


# ============================================================
# 14. Finalize summary
# ============================================================

def finalize_summary(
    all_results,
):
    """
    Calculate mean/std/min/max statistics.
    """

    for activation in all_results:

        for hidden in all_results[
            activation
        ]:

            for k in all_results[
                activation
            ][hidden]:

                group = all_results[
                    activation
                ][hidden][k]

                losses = np.asarray(
                    group["losses"],
                    dtype=np.float64,
                )

                times = np.asarray(
                    group[
                        "training_times_sec"
                    ],
                    dtype=np.float64,
                )

                allocated = np.asarray(
                    group[
                        "peak_memory_allocated_bytes"
                    ],
                    dtype=np.float64,
                )

                reserved = np.asarray(
                    group[
                        "peak_memory_reserved_bytes"
                    ],
                    dtype=np.float64,
                )

                group[
                    "mean_loss"
                ] = float(
                    losses.mean()
                )

                group[
                    "std_loss"
                ] = float(
                    losses.std()
                )

                group[
                    "min_loss"
                ] = float(
                    losses.min()
                )

                group[
                    "max_loss"
                ] = float(
                    losses.max()
                )

                group[
                    "mean_training_time_sec"
                ] = float(
                    times.mean()
                )

                group[
                    "std_training_time_sec"
                ] = float(
                    times.std()
                )

                group[
                    "mean_peak_gpu_memory_allocated_gb"
                ] = float(
                    allocated.mean()
                    / (1024 ** 3)
                )

                group[
                    "max_peak_gpu_memory_allocated_gb"
                ] = float(
                    allocated.max()
                    / (1024 ** 3)
                )

                group[
                    "mean_peak_gpu_memory_reserved_gb"
                ] = float(
                    reserved.mean()
                    / (1024 ** 3)
                )

                group[
                    "max_peak_gpu_memory_reserved_gb"
                ] = float(
                    reserved.max()
                    / (1024 ** 3)
                )


# ============================================================
# 15. Visualization
# ============================================================

def visualize_results(
    all_results,
    hidden_widths,
    fixed_lists,
    save_dir,
    total_time,
):
    """
    Generate comparison plots for all three activations.
    """

    print(
        "\nGenerating summary plots..."
    )

    # ========================================================
    # Plot 1
    # Mean loss vs P (trainable ratio)
    #
    # One figure for each activation.
    # ========================================================

    for activation in CONFIG[
        "activations"
    ]:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        for hidden in hidden_widths:

            P_values = []

            mean_losses = []

            std_losses = []

            for k in fixed_lists[
                hidden
            ]:

                data = all_results[
                    activation
                ][hidden][k]

                P_values.append(
                    data["P"]
                )

                mean_losses.append(
                    data["mean_loss"]
                )

                std_losses.append(
                    data["std_loss"]
                )

            ax.errorbar(
                P_values,
                mean_losses,
                yerr=std_losses,
                fmt="o-",
                capsize=4,
                linewidth=2,
                markersize=5,
                label=(
                    f"Hidden width = {hidden}"
                ),
            )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            "Trainable Neuron Ratio P"
        )

        ax.set_ylabel(
            "Best Training Loss (mean)"
        )

        ax.set_title(
            f"Best Training Loss vs Trainable Neuron Ratio\n"
            f"5 Hidden Layers, Width Scaling, "
            f"{activation.upper()}, 38 Repeats"
        )

        ax.set_xlim(
            -0.03,
            1.03,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                (
                    f"loss_vs_P_"
                    f"{activation}.png"
                ),
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # Plot 2
    # Width vs loss at each P
    # ========================================================

    for activation in CONFIG[
        "activations"
    ]:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        for P_index, P in enumerate(
            CONFIG[
                "fixed_ratio_values"
            ]
        ):

            losses = []

            for hidden in hidden_widths:

                k = fixed_lists[
                    hidden
                ][P_index]

                losses.append(
                    all_results[
                        activation
                    ][hidden][k][
                        "mean_loss"
                    ]
                )

            ax.plot(
                hidden_widths,
                losses,
                "o-",
                linewidth=2,
                markersize=5,
                label=(
                    f"P = {P * 100:.2f}%"
                ),
            )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            "Hidden Width"
        )

        ax.set_ylabel(
            "Best Training Loss (mean)"
        )

        ax.set_title(
            f"Best Training Loss vs Hidden Width\n"
            f"5 Hidden Layers, "
            f"{activation.upper()}, 38 Repeats"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend(
            ncol=3,
            fontsize=9,
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                (
                    f"loss_vs_width_"
                    f"{activation}.png"
                ),
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # Plot 3
    # Activation comparison at each width
    #
    # Each figure contains:
    #   ReLU
    #   GeLU
    #   Tanh
    # ========================================================

    for hidden in hidden_widths:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        for activation in CONFIG[
            "activations"
        ]:

            P_values = []

            mean_losses = []

            for k in fixed_lists[
                hidden
            ]:

                data = all_results[
                    activation
                ][hidden][k]

                P_values.append(
                    data["P"]
                )

                mean_losses.append(
                    data["mean_loss"]
                )

            ax.plot(
                P_values,
                mean_losses,
                "o-",
                linewidth=2,
                markersize=5,
                label=activation.upper(),
            )

        ax.set_yscale(
            "log"
        )

        ax.set_xlabel(
            "Trainable Neuron Ratio P"
        )

        ax.set_ylabel(
            "Best Training Loss (mean)"
        )

        ax.set_title(
            f"Activation Comparison\n"
            f"5 Hidden Layers, "
            f"Hidden Width = {hidden}, "
            f"38 Repeats"
        )

        ax.set_xlim(
            -0.03,
            1.03,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                (
                    f"activation_comparison_"
                    f"width_{hidden}.png"
                ),
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # Plot 4
    # Direct activation comparison
    #
    # Normalize each activation by its P=0 loss
    # for easier relative comparison.
    # ========================================================

    for hidden in hidden_widths:

        fig, ax = plt.subplots(
            figsize=(12, 8)
        )

        for activation in CONFIG[
            "activations"
        ]:

            baseline_k = (
                fixed_lists[
                    hidden
                ][0]
            )

            baseline_loss = (
                all_results[
                    activation
                ][hidden][
                    baseline_k
                ][
                    "mean_loss"
                ]
            )

            P_values = []

            relative_losses = []

            for k in fixed_lists[
                hidden
            ]:

                mean_loss = (
                    all_results[
                        activation
                    ][hidden][k][
                        "mean_loss"
                    ]
                )

                P_values.append(
                    all_results[
                        activation
                    ][hidden][k][
                        "P"
                    ]
                )

                relative_losses.append(
                    mean_loss
                    / baseline_loss
                )

            ax.plot(
                P_values,
                relative_losses,
                "o-",
                linewidth=2,
                markersize=5,
                label=activation.upper(),
            )

        ax.axhline(
            1.0,
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel(
            "Trainable Neuron Ratio P"
        )

        ax.set_ylabel(
            "Relative Mean Loss "
            "(Loss / Loss at P = 0)"
        )

        ax.set_title(
            f"Relative Loss vs Trainable Ratio\n"
            f"5 Hidden Layers, "
            f"Hidden Width = {hidden}"
        )

        ax.set_xlim(
            -0.03,
            1.03,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                (
                    f"relative_loss_"
                    f"width_{hidden}.png"
                ),
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ========================================================
    # Human-readable summary
    # ========================================================

    summary_txt = os.path.join(
        save_dir,
        "detailed_results.txt",
    )

    with open(
        summary_txt,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Fixed-depth hybrid-network "
            "width-scaling experiment\n"
        )

        file.write(
            "=" * 110
            + "\n"
        )

        file.write(
            f"Total training time: "
            f"{total_time:.2f} sec\n"
        )

        file.write(
            f"Hidden depth: "
            f"{CONFIG['depth']}\n"
        )

        file.write(
            f"Hidden widths: "
            f"{hidden_widths}\n"
        )

        file.write(
            f"Activations: "
            f"{CONFIG['activations']}\n"
        )

        file.write(
            f"Repeats: "
            f"{CONFIG['repeats']}\n"
        )

        file.write(
            "Trainable ratios: "
        )

        file.write(
            ", ".join(
                f"{P * 100:.2f}%"
                for P in CONFIG[
                    "fixed_ratio_values"
                ]
            )
        )

        file.write(
            "\n"
        )

        file.write(
            f"Dataset: "
            f"{CONFIG['target_formula']}\n"
        )

        file.write(
            f"x range: "
            f"[-2*pi, 2*pi]\n"
        )

        file.write(
            f"Dataset size: "
            f"{CONFIG['data_amount']}\n"
        )

        file.write(
            "Initialization: "
            "same neuron-wise Xavier pool "
            "for all P within each "
            "(activation, width, repeat)\n"
        )

        file.write(
            "Data order: deterministic "
            "np.linspace order; no shuffle\n"
        )

        file.write(
            "P meaning: fraction of "
            "trainable neurons per hidden layer\n"
        )

        file.write(
            "=" * 110
            + "\n\n"
        )

        # ----------------------------------------------------
        # Detailed result by activation
        # ----------------------------------------------------

        for activation in CONFIG[
            "activations"
        ]:

            file.write(
                f"Activation = "
                f"{activation.upper()}\n"
            )

            file.write(
                "-" * 110
                + "\n"
            )

            for hidden in hidden_widths:

                file.write(
                    f"\nHidden width = "
                    f"{hidden}\n"
                )

                file.write(
                    "-" * 90
                    + "\n"
                )

                for k in fixed_lists[
                    hidden
                ]:

                    data = all_results[
                        activation
                    ][hidden][k]

                    file.write(
                        f"k={k:3d}, "
                        f"P={data['P']:.4f}, "
                        f"mean_loss="
                        f"{data['mean_loss']:.6e}, "
                        f"std="
                        f"{data['std_loss']:.6e}, "
                        f"min="
                        f"{data['min_loss']:.6e}, "
                        f"max="
                        f"{data['max_loss']:.6e}, "
                        f"mean_time="
                        f"{data['mean_training_time_sec']:.2f}s\n"
                    )

                file.write(
                    "\n"
                )

            file.write(
                "\n"
            )

    # ========================================================
    # Machine-readable summary JSON
    # ========================================================

    json_summary = {}

    for activation in CONFIG[
        "activations"
    ]:

        json_summary[
            activation
        ] = {}

        for hidden in hidden_widths:

            json_summary[
                activation
            ][str(hidden)] = {}

            for k in fixed_lists[
                hidden
            ]:

                data = all_results[
                    activation
                ][hidden][k]

                json_summary[
                    activation
                ][str(hidden)][
                    str(k)
                ] = {

                    "P": data[
                        "P"
                    ],

                    "mean_loss": data[
                        "mean_loss"
                    ],

                    "std_loss": data[
                        "std_loss"
                    ],

                    "min_loss": data[
                        "min_loss"
                    ],

                    "max_loss": data[
                        "max_loss"
                    ],

                    "mean_training_time_sec": (
                        data[
                            "mean_training_time_sec"
                        ]
                    ),

                    "std_training_time_sec": (
                        data[
                            "std_training_time_sec"
                        ]
                    ),

                    "mean_peak_gpu_memory_allocated_gb": (
                        data[
                            "mean_peak_gpu_memory_allocated_gb"
                        ]
                    ),

                    "max_peak_gpu_memory_allocated_gb": (
                        data[
                            "max_peak_gpu_memory_allocated_gb"
                        ]
                    ),

                    "mean_peak_gpu_memory_reserved_gb": (
                        data[
                            "mean_peak_gpu_memory_reserved_gb"
                        ]
                    ),

                    "max_peak_gpu_memory_reserved_gb": (
                        data[
                            "max_peak_gpu_memory_reserved_gb"
                        ]
                    ),

                    "repeats": data[
                        "repeats"
                    ],
                }

    with open(
        os.path.join(
            save_dir,
            "summary.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSummary files and plots saved to:"
        f"\n{save_dir}"
    )


# ============================================================
# 16. Main experiment
# ============================================================

def train_width_scaling_experiment():
    """
    Run all activation functions.
    """

    print(
        "=" * 120
    )

    print(
        "Fixed-depth hybrid-network "
        "width-scaling experiment"
    )

    print(
        "Activations: ReLU / GeLU / Tanh"
    )

    print(
        "5 hidden layers"
    )

    print(
        "Widths: 32 / 64 / 128 / 256"
    )

    print(
        "Trainable ratios: 0% -> 100%"
    )

    print(
        "Dense sampling in the first 50%"
    )

    print(
        "38 repeats"
    )

    print(
        "Target: sin(5x) + 0.5*cos(9x) + "
        "0.35*sin(13x-0.7) + "
        "0.2*cos(17x+0.4) + "
        "0.1*sin(21x)"
    )

    print(
        "Domain: [-2*pi, 2*pi]"
    )

    print(
        "=" * 120
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        print(
            "Error: CUDA is not available."
        )

        return None

    gpu_index = CONFIG[
        "gpu_index"
    ]

    if (
        gpu_index
        >= torch.cuda.device_count()
    ):

        print(
            f"Error: requested GPU "
            f"{gpu_index} is unavailable. "
            f"Visible GPU count: "
            f"{torch.cuda.device_count()}"
        )

        return None

    torch.cuda.set_device(
        gpu_index
    )

    gpu_name = (
        torch.cuda.get_device_name(
            gpu_index
        )
    )

    gpu_memory_gb = (
        torch.cuda.get_device_properties(
            gpu_index
        ).total_memory
        / (1024 ** 3)
    )

    # --------------------------------------------------------
    # Experiment parameters
    # --------------------------------------------------------

    activations = CONFIG[
        "activations"
    ]

    hidden_widths = CONFIG[
        "hidden_widths"
    ]

    P_values = CONFIG[
        "fixed_ratio_values"
    ]

    repeats = CONFIG[
        "repeats"
    ]

    print(
        f"GPU: {gpu_index} - "
        f"{gpu_name} "
        f"({gpu_memory_gb:.1f} GB)"
    )

    print(
        f"Hidden depth: "
        f"{CONFIG['depth']}"
    )

    print(
        f"Hidden widths: "
        f"{hidden_widths}"
    )

    print(
        "Activations: "
        + ", ".join(
            activation.upper()
            for activation
            in activations
        )
    )

    print(
        "Trainable ratios: "
        + ", ".join(
            f"{P * 100:.2f}%"
            for P in P_values
        )
    )

    print(
        f"Repeats: {repeats}"
    )

    print(
        f"Dataset size: "
        f"{CONFIG['data_amount']}"
    )

    print(
        "Dataset domain: "
        "[-2*pi, 2*pi]"
    )

    print(
        f"Epochs: "
        f"{CONFIG['epochs']} "
        "(no early stopping)"
    )

    print(
        f"CPU worker processes: "
        f"{CONFIG['max_workers']}"
    )

    print(
        "Maximum simultaneous GPU tasks: "
        f"{CONFIG['max_concurrent_gpu_tasks']}"
    )

    print(
        "=" * 120
    )

    # --------------------------------------------------------
    # Calculate exact k values
    #
    # k is now the number of TRAINABLE neurons per layer.
    # --------------------------------------------------------

    fixed_lists = {}

    for hidden in hidden_widths:

        fixed_lists[
            hidden
        ] = [
            int(
                round(
                    P * hidden
                )
            )
            for P in P_values
        ]

        # Verify exact representability
        actual_P = [
            k / hidden
            for k in fixed_lists[
                hidden
            ]
        ]

        for requested, actual in zip(
            P_values,
            actual_P,
        ):

            if not np.isclose(
                requested,
                actual,
                atol=1e-12,
            ):

                raise RuntimeError(
                    f"P={requested} is not exactly "
                    f"representable for hidden={hidden}"
                )

        print(
            f"hidden={hidden}: "
            f"trainable_k={fixed_lists[hidden]}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_dir = (
        "results_fixed_depth5_"
        "width_scaling_"
        "relu_gelu_tanh_"
        "new_target_"
        "trainable_ratio_"
        "38repeats_"
        f"{timestamp}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config_to_save = dict(
        CONFIG
    )

    total_tasks = (
        len(activations)
        * len(hidden_widths)
        * len(P_values)
        * repeats
    )

    config_to_save.update(
        {
            "timestamp": timestamp,

            "save_dir": save_dir,

            "trainable_neuron_lists": (
                fixed_lists
            ),

            "total_tasks": (
                total_tasks
            ),

            "gpu_name": gpu_name,

            "gpu_memory_gb": (
                gpu_memory_gb
            ),

            "network_architecture": (
                "shared-input hybrid MLP; "
                "fixed and trainable neuron "
                "groups both receive the "
                "complete previous-layer output"
            ),

            "initialization_policy": (
                "one complete neuron-wise "
                "Xavier pool per "
                "(activation, hidden_width, repeat); "
                "identical across P"
            ),

            "data_order_policy": (
                "deterministic np.linspace order; "
                "no shuffle"
            ),

            "ratio_sampling_policy": (
                "trainable-ratio grid is denser "
                "in the first 50% and returns "
                "to 12.5% spacing after 50%"
            ),

            "ratio_meaning": (
                "P = fraction of trainable "
                "neurons per hidden layer; "
                "k = round(P * hidden) trainable, "
                "(hidden - k) fixed"
            ),
        }
    )

    with open(
        os.path.join(
            save_dir,
            "experiment_config.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_to_save,
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        os.path.join(
            save_dir,
            "experiment_config.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            config_to_save,
            file,
        )

    with open(
        os.path.join(
            save_dir,
            "experiment_config.txt",
        ),
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "Fixed-depth hybrid-network "
            "width-scaling experiment\n"
        )

        file.write(
            "=" * 100
            + "\n"
        )

        for key, value in (
            config_to_save.items()
        ):

            file.write(
                f"{key}: {value}\n"
            )

    # --------------------------------------------------------
    # Adam configuration
    # --------------------------------------------------------

    adam_cfg = {

        "epochs": CONFIG[
            "epochs"
        ],

        "lr": CONFIG[
            "lr"
        ],

        "min_lr": CONFIG[
            "min_lr"
        ],

        "lr_factor": CONFIG[
            "lr_factor"
        ],

        "lr_patience": CONFIG[
            "lr_patience"
        ],

        "lr_threshold": CONFIG[
            "lr_threshold"
        ],

        "print_every": CONFIG[
            "print_every"
        ],

        "log_every": CONFIG[
            "log_every"
        ],

        "gradient_clip_norm": (
            CONFIG[
                "gradient_clip_norm"
            ]
        ),
    }

    # --------------------------------------------------------
    # Build tasks
    #
    # Each task tuple still uses "k", but k now means
    # the number of TRAINABLE neurons per hidden layer.
    # --------------------------------------------------------

    tasks = []

    task_id = 0

    for activation in activations:

        for hidden in hidden_widths:

            for k in fixed_lists[
                hidden
            ]:

                for repeat_idx in range(
                    repeats
                ):

                    tasks.append(
                        (
                            activation,
                            hidden,
                            k,
                            repeat_idx,
                            adam_cfg,
                            task_id,
                            save_dir,
                        )
                    )

                    task_id += 1

    print(
        f"\nTotal tasks: "
        f"{len(tasks)}"
    )

    print(
        f"Calculation: "
        f"{len(activations)} activations "
        f"x {len(hidden_widths)} widths "
        f"x {len(P_values)} P values "
        f"x {repeats} repeats "
        f"= {len(tasks)} tasks"
    )

    # --------------------------------------------------------
    # Initialize aggregate storage
    # --------------------------------------------------------

    all_results = {
        activation: {}
        for activation in activations
    }

    completed = 0

    global_start_time = time.time()

    # --------------------------------------------------------
    # GPU concurrency
    # --------------------------------------------------------

    batch_size = CONFIG[
        "max_concurrent_gpu_tasks"
    ]

    num_batches = (
        len(tasks)
        + batch_size
        - 1
    ) // batch_size

    # --------------------------------------------------------
    # Process pool
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=CONFIG[
            "max_workers"
        ]
    ) as executor:

        for batch_idx in range(
            num_batches
        ):

            start_idx = (
                batch_idx
                * batch_size
            )

            end_idx = min(
                start_idx
                + batch_size,
                len(tasks),
            )

            batch_tasks = tasks[
                start_idx:end_idx
            ]

            print(
                f"\nSubmitting batch "
                f"{batch_idx + 1}/"
                f"{num_batches}: "
                f"{len(batch_tasks)} tasks"
            )

            futures = {
                executor.submit(
                    train_single_task_single_gpu,
                    task,
                ): task

                for task in batch_tasks
            }

            # ------------------------------------------------
            # Collect results
            # ------------------------------------------------

            for future in as_completed(
                futures
            ):

                try:

                    result = (
                        future.result()
                    )

                except Exception as error:

                    failed_task = futures[
                        future
                    ]

                    print(
                        "\nERROR in task:"
                    )

                    print(
                        failed_task
                    )

                    raise error

                # --------------------------------------------
                # Save run
                # --------------------------------------------

                save_run_result(
                    result,
                    save_dir,
                )

                # --------------------------------------------
                # Update summary
                # --------------------------------------------

                update_summary(
                    all_results,
                    result,
                )

                completed += 1

                elapsed = (
                    time.time()
                    - global_start_time
                )

                # --------------------------------------------
                # Progress report
                # --------------------------------------------

                if (
                    completed == 1
                    or completed % 20 == 0
                    or completed
                    == len(tasks)
                ):

                    avg_time = (
                        elapsed
                        / completed
                    )

                    remaining = (
                        avg_time
                        * (
                            len(tasks)
                            - completed
                        )
                    )

                    print(
                        f"\nProgress: "
                        f"{completed}/"
                        f"{len(tasks)} "
                        f"("
                        f"{100.0 * completed / len(tasks):.2f}%"
                        f")"
                    )

                    print(
                        f"Elapsed: "
                        f"{elapsed / 60:.1f} min | "
                        f"Estimated remaining: "
                        f"{remaining / 60:.1f} min"
                    )

                    print(
                        f"Latest: "
                        f"{result['activation'].upper()}, "
                        f"hidden={result['hidden']}, "
                        f"k={result['k']}, "
                        f"P={result['P']:.4f}, "
                        f"repeat={result['repeat_idx']}, "
                        f"best_loss="
                        f"{result['best_loss']:.6e}"
                    )

                    # ----------------------------------------
                    # Intermediate checkpoint
                    # ----------------------------------------

                    checkpoint_path = os.path.join(
                        save_dir,
                        (
                            f"intermediate_summary_"
                            f"{completed}.pkl"
                        ),
                    )

                    with open(
                        checkpoint_path,
                        "wb",
                    ) as file:

                        pickle.dump(
                            {
                                "completed": (
                                    completed
                                ),

                                "total_tasks": (
                                    len(tasks)
                                ),

                                "elapsed_sec": (
                                    elapsed
                                ),

                                "all_results": (
                                    all_results
                                ),
                            },
                            file,
                        )

    # --------------------------------------------------------
    # Final timing
    # --------------------------------------------------------

    total_time = (
        time.time()
        - global_start_time
    )

    # --------------------------------------------------------
    # Finalize summary
    # --------------------------------------------------------

    finalize_summary(
        all_results
    )

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    with open(
        os.path.join(
            save_dir,
            "all_results.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            {
                "all_results": (
                    all_results
                ),

                "activations": (
                    activations
                ),

                "hidden_widths": (
                    hidden_widths
                ),

                "fixed_lists": (
                    fixed_lists
                ),

                "total_time_sec": (
                    total_time
                ),

                "completed": (
                    completed
                ),
            },
            file,
        )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "All training tasks completed."
    )

    print(
        f"Total time: "
        f"{total_time:.2f} sec "
        f"("
        f"{total_time / 60:.2f} min"
        f")"
    )

    print(
        f"Average time per task: "
        f"{total_time / len(tasks):.2f} sec"
    )

    print(
        f"Results saved to: "
        f"{save_dir}"
    )

    print(
        "=" * 120
    )

    return (
        all_results,
        activations,
        hidden_widths,
        fixed_lists,
        save_dir,
        total_time,
    )


# ============================================================
# 17. Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "Hybrid-network experiment:"
    )

    print(
        "ReLU + GeLU + Tanh"
    )

    print(
        "5 hidden layers"
    )

    print(
        "Widths = 32, 64, 128, 256"
    )

    print(
        "Dense trainable-ratio sampling in first 50%"
    )

    print(
        "38 repeats"
    )

    print(
        "Target function:"
    )

    print(
        "sin(5x) + 0.5*cos(9x) + "
        "0.35*sin(13x-0.7) + "
        "0.2*cos(17x+0.4) + "
        "0.1*sin(21x)"
    )

    print(
        "Domain: [-2*pi, 2*pi]"
    )

    print(
        "=" * 120
    )

    result = (
        train_width_scaling_experiment()
    )

    if result is None:

        print(
            "Experiment failed."
        )

        return

    (
        all_results,
        activations,
        hidden_widths,
        fixed_lists,
        save_dir,
        total_time,
    ) = result

    # --------------------------------------------------------
    # Generate plots and summaries
    # --------------------------------------------------------

    visualize_results(
        all_results=all_results,

        hidden_widths=hidden_widths,

        fixed_lists=fixed_lists,

        save_dir=save_dir,

        total_time=total_time,
    )

    print(
        "\nExperiment completed successfully."
    )

    print(
        f"Results directory: "
        f"{save_dir}"
    )


# ============================================================
# 18. Entry point
# ============================================================

if __name__ == "__main__":

    mp.set_start_method(
        "spawn",
        force=True,
    )

    main()