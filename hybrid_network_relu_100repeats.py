# Single-GPU version: Investigating the Overall Impact of Different Numbers of Hidden Layers (Activation Function: ReLU)
# Q = 12000
# Number of hidden layers = 1-10
# Hybrid network architecture, hidden layer size = 128
# The fixed and trainable components of each layer both receive the full output of the previous layer


import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import random
import time
import os
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ============================================================
# 1. Set Random Seed
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# ============================================================
# 2. Construct 1D Dataset
# ============================================================
def build_1d_dataset(n=12000):
    """Generate 1D dataset with new target function for testing universality"""
    x = np.linspace(-2 * np.pi, 2 * np.pi, n).reshape(-1, 1)
    y = (
        np.sin(2 * x) + 
        np.cos(5 * x) + 
        0.3 * np.sin(8 * x) + 
        0.2 * np.cos(11 * x) + 
        0.1 * np.sin(13 * x)
    )
    return x.astype(np.float32), y.astype(np.float32)

# ============================================================
# 3. Activation Function Factory (Tanh, ReLU, GELU)
# ============================================================
def get_activation(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {name}")

# ============================================================
# 4. Xavier Initialization Tools
# ============================================================
def xavier_tensor(shape):
    t = torch.empty(shape, dtype=torch.float32)
    nn.init.xavier_uniform_(t)
    return t

def zeros_tensor(shape):
    return torch.zeros(shape, dtype=torch.float32)

# ============================================================
# 5. Create Uniform Initialization Parameter Pool
# ============================================================
def create_uniform_init_pool(hidden=128, input_dim=1, layers=4, seed=42):
    set_seed(seed)
    
    init_pool = {}
    
    init_pool["l1_fixed_w"] = xavier_tensor((hidden, input_dim))
    init_pool["l1_fixed_b"] = zeros_tensor((hidden,))
    init_pool["l1_train_w"] = xavier_tensor((hidden, input_dim))
    init_pool["l1_train_b"] = zeros_tensor((hidden,))
    
    for i in range(2, layers + 1):
        init_pool[f"l{i}_fixed_w"] = xavier_tensor((hidden, hidden))
        init_pool[f"l{i}_fixed_b"] = zeros_tensor((hidden,))
        init_pool[f"l{i}_train_w"] = xavier_tensor((hidden, hidden))
        init_pool[f"l{i}_train_b"] = zeros_tensor((hidden,))
    
    init_pool["out_w"] = xavier_tensor((1, hidden))
    init_pool["out_b"] = zeros_tensor((1,))
    
    return init_pool

# ============================================================
# 6. Hybrid Layer
# ============================================================
class HybridLayer(nn.Module):
    def __init__(self, in_dim, out_fixed, out_train, activation="relu"):
        super().__init__()
        self.act = get_activation(activation)
        
        self.fixed_linear = nn.Linear(in_dim, out_fixed) if out_fixed > 0 else None
        self.train_linear = nn.Linear(in_dim, out_train) if out_train > 0 else None
        
        if self.fixed_linear is not None:
            for p in self.fixed_linear.parameters():
                p.requires_grad = False

    def forward(self, x):
        fixed_out = self.act(self.fixed_linear(x)) if self.fixed_linear is not None else None
        train_out = self.act(self.train_linear(x)) if self.train_linear is not None else None
        return fixed_out, train_out

# ============================================================
# 7. Hybrid Network
# ============================================================
class HybridNet1D(nn.Module):
    def __init__(self, input_dim=1, hidden=128, fixed_k=64, layers=4, activation="relu"):
        super().__init__()
        assert layers >= 1, "layers must be >= 1"
        
        self.fixed_k = fixed_k
        self.train_k = hidden - fixed_k
        self.hidden = hidden
        self.num_layers = layers
        self.layers = nn.ModuleList()
        
        self.layers.append(
            HybridLayer(
                in_dim=input_dim,
                out_fixed=fixed_k,
                out_train=self.train_k,
                activation=activation
            )
        )
        
        for _ in range(layers - 1):
            self.layers.append(
                HybridLayer(
                    in_dim=hidden,
                    out_fixed=fixed_k,
                    out_train=self.train_k,
                    activation=activation
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
# 8. Uniform Initialization
# ============================================================
def apply_uniform_init_to_model(model, init_pool):
    k = model.fixed_k
    t = model.train_k
    hidden = model.hidden
    layers = model.num_layers
    
    with torch.no_grad():
        layer1 = model.layers[0]
        if layer1.fixed_linear is not None and k > 0:
            layer1.fixed_linear.weight.copy_(init_pool["l1_fixed_w"][:k, :])
            layer1.fixed_linear.bias.copy_(init_pool["l1_fixed_b"][:k])
        if layer1.train_linear is not None and t > 0:
            layer1.train_linear.weight.copy_(init_pool["l1_train_w"][k:, :])
            layer1.train_linear.bias.copy_(init_pool["l1_train_b"][k:])
        
        for i in range(2, layers + 1):
            layer = model.layers[i - 1]
            if layer.fixed_linear is not None and k > 0:
                layer.fixed_linear.weight.copy_(init_pool[f"l{i}_fixed_w"][:k, :])
                layer.fixed_linear.bias.copy_(init_pool[f"l{i}_fixed_b"][:k])
            if layer.train_linear is not None and t > 0:
                layer.train_linear.weight.copy_(init_pool[f"l{i}_train_w"][k:, :])
                layer.train_linear.bias.copy_(init_pool[f"l{i}_train_b"][k:])
        
        model.out.weight.copy_(init_pool["out_w"])
        model.out.bias.copy_(init_pool["out_b"])

# ============================================================
# 9. Single GPU Training Function
# ============================================================
def train_single_experiment_a100(args):
    """Run experiment on a single A100 - NO EARLY STOPPING"""
    layer_num, k, repeat_idx, init_pool, adam_cfg, task_id = args
    
    device = torch.device("cuda:0")
    set_seed(42 + repeat_idx * 1000 + layer_num * 100 + k)
    
    task_start_time = time.time()
    
    x, y = build_1d_dataset(n=12000)
    x_train = torch.tensor(x, dtype=torch.float32).to(device)
    y_train = torch.tensor(y, dtype=torch.float32).to(device)
    
    model = HybridNet1D(
        input_dim=1,
        hidden=128,
        fixed_k=k,
        layers=layer_num,
        activation="relu"
    ).to(device)
    
    apply_uniform_init_to_model(model, init_pool)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_cfg["lr"])
    loss_fn = nn.MSELoss()
    
    epochs = adam_cfg["epochs"]
    print_every = adam_cfg["print_every"]
    
    best_loss = float("inf")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        if loss_value < best_loss:
            best_loss = loss_value
        
        if (epoch + 1) % print_every == 0:
            print(f"  Task {task_id}: L={layer_num}, k={k}, r={repeat_idx}, "
                  f"Epoch {epoch+1}/{epochs}, loss={loss_value:.6e}, best={best_loss:.6e}")
    
    task_end_time = time.time()
    task_duration = task_end_time - task_start_time
    
    del model, x_train, y_train
    torch.cuda.empty_cache()
    
    return layer_num, k, repeat_idx, best_loss, task_duration, epochs

# ============================================================
# 10. Single A100 Parallel Training Main Function
# ============================================================
def run_single_gpu_experiment():
    """Single A100 parallel training main function"""
    print("=" * 120)
    print("Hybrid Network Experiment with Variable Layers - 1 A100 GPU + 16 CPU Cores Parallel")
    print("Network Design: Fixed and trainable parts receive entire output from previous layer")
    print("Activation Function: RELU")
    print("=" * 120)
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available!")
        return None, None, None, None
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    hidden = 128
    layer_list = list(range(1, 11))
    
    fixed_list = []
    fixed_list.extend(range(0, 65, 8))
    fixed_list.extend(range(70, 113, 6))
    fixed_list.extend(range(116, 129, 4))
    fixed_list = sorted(list(set(fixed_list)))
    
    activation = "relu"
    repeats = 100
    base_seed = 42
    
    adam_cfg = {
        "epochs": 12000,
        "lr": 1e-3,
        "patience": 600,
        "print_every": 600
    }
    
    print(f"\nHidden layers: {layer_list}")
    print(f"Network architecture: 1-128-...-128-1 (variable hidden layers)")
    print(f"Network design: Fixed and trainable parts receive entire output from previous layer")
    print(f"Activation function: {activation}")
    print(f"Optimizer: Adam, {adam_cfg['epochs']} epochs (NO EARLY STOPPING)")
    print(f"Number of data points: 12000")
    print(f"Target function: sin(2x) + cos(5x) + 0.3*sin(8x) + 0.2*cos(11x) + 0.1*sin(13x)")
    print(f"Number of k values analyzed: {len(fixed_list)}")
    print(f"Repetitions per configuration: {repeats}")
    print(f"Total experiments: {len(layer_list)} × {len(fixed_list)} × {repeats} = {len(layer_list)*len(fixed_list)*repeats}")
    print("=" * 120)
    
    print("Creating initialization parameter pool...")
    init_pools = {}
    for layer_num in layer_list:
        init_pools[layer_num] = []
        for r in range(repeats):
            init_pool = create_uniform_init_pool(
                hidden=hidden,
                input_dim=1,
                layers=layer_num,
                seed=base_seed + r + layer_num * 100
            )
            init_pools[layer_num].append(init_pool)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"results_single_gpu_Q12000_no_early_stop_newfunc_relu_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results directory: {save_dir}")
    print(f"Starting 1 GPU parallel training, using {num_gpus} GPU(s), 16 CPU cores...")
    
    tasks = []
    task_id = 0
    for layer_idx, layer_num in enumerate(layer_list):
        for k_idx, k in enumerate(fixed_list):
            for r in range(repeats):
                tasks.append((layer_num, k, r, init_pools[layer_num][r], 
                            adam_cfg, task_id))
                task_id += 1
    
    print(f"\nPrepared {len(tasks)} tasks")
    print("Using 16 CPU cores to process tasks in parallel...")
    
    all_results = {}
    for layer_num in layer_list:
        all_results[layer_num] = {}
        for k in fixed_list:
            all_results[layer_num][k] = {
                'losses': [],
                'durations': [],
                'epochs': []
            }
    
    global_start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(train_single_experiment_a100, task): task for task in tasks}
        
        completed = 0
        total_tasks = len(tasks)
        start_time = time.time()
        last_print_time = time.time()
        
        for future in as_completed(futures):
            layer_num, k, repeat_idx, best_loss, task_duration, epochs_used = future.result()
            
            all_results[layer_num][k]['losses'].append(best_loss)
            all_results[layer_num][k]['durations'].append(task_duration)
            all_results[layer_num][k]['epochs'].append(epochs_used)
            
            completed += 1
            
            current_time = time.time()
            if completed % 10 == 0 or (current_time - last_print_time) > 60 or completed == total_tasks:
                elapsed = current_time - start_time
                avg_time_per_task = elapsed / completed
                remaining_tasks = total_tasks - completed
                remaining_time = avg_time_per_task * remaining_tasks / 60
                elapsed_total = current_time - global_start_time
                
                print(f"\nProgress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                print(f"Elapsed: {elapsed/60:.1f} minutes, Avg per task: {avg_time_per_task:.1f} seconds")
                print(f"Estimated remaining: {remaining_time:.1f} minutes, Total: {elapsed_total/60:.1f} minutes")
                
                last_print_time = current_time
    
    total_time = time.time() - global_start_time
    
    for layer_num in layer_list:
        for k in fixed_list:
            if all_results[layer_num][k]['losses']:
                losses = np.array(all_results[layer_num][k]['losses'])
                durations = np.array(all_results[layer_num][k]['durations'])
                
                all_results[layer_num][k]['mean_loss'] = losses.mean()
                all_results[layer_num][k]['std_loss'] = losses.std()
                all_results[layer_num][k]['min_loss'] = losses.min()
                all_results[layer_num][k]['max_loss'] = losses.max()
                
                all_results[layer_num][k]['mean_duration'] = durations.mean()
                all_results[layer_num][k]['std_duration'] = durations.std()
                all_results[layer_num][k]['min_duration'] = durations.min()
                all_results[layer_num][k]['max_duration'] = durations.max()
                
                all_results[layer_num][k]['P'] = k / hidden
    
    print(f"\n{'='*120}")
    print(f"All tasks completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average per task: {total_time/len(tasks):.2f} seconds")
    print(f"Results directory: {save_dir}")
    print(f"{'='*120}")
    
    return all_results, layer_list, fixed_list, save_dir, total_time

# ============================================================
# 11. Save Results to TXT File
# ============================================================
def save_results_to_txt(all_results, layer_list, fixed_list, save_dir, total_time):
    print("Saving results to TXT file...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_file = os.path.join(save_dir, 'experiment_results.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("HYBRID NETWORK EXPERIMENT RESULTS (Q=12000) - NEW TARGET FUNCTION - ACTIVATION: RELU\n")
        f.write("=" * 120 + "\n\n")
        
        f.write(f"Experiment time: {timestamp}\n")
        f.write(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Number of data points: 12000\n")
        f.write(f"Target function: sin(2x) + cos(5x) + 0.3*sin(8x) + 0.2*cos(11x) + 0.1*sin(13x)\n")
        f.write(f"Repetitions per configuration: 100\n")
        f.write(f"Hidden layers range: {layer_list}\n")
        f.write(f"Network architecture: 1-128-...-128-1 (variable hidden layers)\n")
        f.write(f"Activation function: RELU\n")
        f.write(f"Optimizer: Adam, 12000 epochs (NO EARLY STOPPING)\n")
        f.write(f"Network design: Fixed and trainable parts receive entire output from previous layer\n")
        f.write(f"Number of GPUs used: 1\n")
        f.write(f"Number of CPU cores used: 16\n\n")
        
        f.write("=" * 120 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 120 + "\n\n")
        
        for layer_num in layer_list:
            f.write(f"\nHidden Layers: {layer_num}\n")
            f.write("-" * 100 + "\n")
            
            best_loss = float('inf')
            best_k = None
            best_duration = None
            
            for k in fixed_list:
                if k in all_results[layer_num] and 'mean_loss' in all_results[layer_num][k]:
                    data = all_results[layer_num][k]
                    f.write(f"  k={k:3d}, P={data['P']:.4f}:\n")
                    f.write(f"    Loss: mean={data['mean_loss']:.6e}, std={data['std_loss']:.2e}, ")
                    f.write(f"min={data['min_loss']:.6e}, max={data['max_loss']:.6e}\n")
                    f.write(f"    Time: mean={data['mean_duration']:.1f}s, std={data['std_duration']:.1f}s, ")
                    f.write(f"min={data['min_duration']:.1f}s, max={data['max_duration']:.1f}s\n")
                    
                    if data['mean_loss'] < best_loss:
                        best_loss = data['mean_loss']
                        best_k = k
                        best_duration = data['mean_duration']
            
            if best_k is not None:
                f.write(f"\n  Best configuration for {layer_num} layers:\n")
                f.write(f"    k={best_k}, P={best_k/128:.4f}\n")
                f.write(f"    Loss: {best_loss:.6e}\n")
                f.write(f"    Time: {best_duration:.1f}s\n")
        
        f.write("\n" + "=" * 120 + "\n")
        f.write("OVERALL BEST CONFIGURATION\n")
        f.write("=" * 120 + "\n\n")
        
        best_overall_loss = float('inf')
        best_overall_layer = None
        best_overall_k = None
        best_overall_data = None
        
        for layer_num in layer_list:
            for k in fixed_list:
                if k in all_results[layer_num] and 'mean_loss' in all_results[layer_num][k]:
                    if all_results[layer_num][k]['mean_loss'] < best_overall_loss:
                        best_overall_loss = all_results[layer_num][k]['mean_loss']
                        best_overall_layer = layer_num
                        best_overall_k = k
                        best_overall_data = all_results[layer_num][k]
        
        if best_overall_data:
            f.write(f"Best overall configuration:\n")
            f.write(f"  Layers: {best_overall_layer}\n")
            f.write(f"  k: {best_overall_k}\n")
            f.write(f"  P: {best_overall_data['P']:.4f}\n")
            f.write(f"  Loss: {best_overall_loss:.6e}\n")
            f.write(f"  Time: {best_overall_data['mean_duration']:.1f}s\n")
    
    print(f"Results saved to: {result_file}")
    return result_file

# ============================================================
# 12. Generate Plots
# ============================================================
def generate_plots(all_results, layer_list, fixed_list, save_dir):
    print("Generating plots...")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Loss vs P for different layers
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_list)))
    for i, layer_num in enumerate(layer_list):
        P_values = []
        mean_losses = []
        for k in fixed_list:
            if k in all_results[layer_num] and 'mean_loss' in all_results[layer_num][k]:
                P_values.append(all_results[layer_num][k]['P'])
                mean_losses.append(all_results[layer_num][k]['mean_loss'])
        if P_values and mean_losses:
            plt.plot(P_values, mean_losses, 'o-', linewidth=1.5, markersize=4, 
                    label=f'{layer_num} Layers', color=colors[i])
    plt.yscale('log')
    plt.xlabel('Fixed Parameter Ratio P', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss vs Fixed Parameter Ratio P for Different Number of Hidden Layers\n(RELU, 12000 points, 100 reps, No early stopping)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10, loc='upper right')
    plt.xlim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/loss_vs_P_by_layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Time vs P for different layers
    plt.figure(figsize=(12, 8))
    for i, layer_num in enumerate(layer_list):
        P_values = []
        mean_times = []
        for k in fixed_list:
            if k in all_results[layer_num] and 'mean_duration' in all_results[layer_num][k]:
                P_values.append(all_results[layer_num][k]['P'])
                mean_times.append(all_results[layer_num][k]['mean_duration'])
        if P_values and mean_times:
            plt.plot(P_values, mean_times, 's-', linewidth=1.5, markersize=4, 
                    label=f'{layer_num} Layers', color=colors[i])
    plt.xlabel('Fixed Parameter Ratio P', fontsize=12)
    plt.ylabel('Training Time (s)', fontsize=12)
    plt.title('Training Time vs Fixed Parameter Ratio P for Different Number of Hidden Layers\n(RELU, 12000 points, 100 reps, No early stopping)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper left')
    plt.xlim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig(f'{save_dir}/time_vs_P_by_layers.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Best loss vs number of layers
    plt.figure(figsize=(10, 6))
    best_losses = []
    best_times = []
    best_Ps = []
    for layer_num in layer_list:
        best_loss = float('inf')
        best_time = 0
        best_P = 0
        for k in fixed_list:
            if k in all_results[layer_num] and 'mean_loss' in all_results[layer_num][k]:
                if all_results[layer_num][k]['mean_loss'] < best_loss:
                    best_loss = all_results[layer_num][k]['mean_loss']
                    best_time = all_results[layer_num][k]['mean_duration']
                    best_P = all_results[layer_num][k]['P']
        if best_loss < float('inf'):
            best_losses.append(best_loss)
            best_times.append(best_time)
            best_Ps.append(best_P)
    
    if best_losses:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(layer_list[:len(best_losses)], best_losses, 'bo-', linewidth=2, markersize=8, label='Best Loss')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Hidden Layers', fontsize=12)
        ax1.set_ylabel('Best Loss', fontsize=12, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3, which='both')
        
        ax2 = ax1.twinx()
        ax2.plot(layer_list[:len(best_times)], best_times, 'r^-', linewidth=2, markersize=8, label='Training Time')
        ax2.set_ylabel('Training Time (s)', fontsize=12, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        plt.title('Best Loss and Training Time vs Number of Hidden Layers\n(RELU, 12000 points, 100 reps)', 
                  fontsize=14, fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/best_loss_time_vs_layers.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Heatmap for loss
    plt.figure(figsize=(12, 8))
    heatmap_data = []
    for layer_num in layer_list:
        row_data = []
        for k in fixed_list:
            if k in all_results[layer_num] and 'mean_loss' in all_results[layer_num][k]:
                row_data.append(all_results[layer_num][k]['mean_loss'])
        if row_data:
            heatmap_data.append(row_data)
    if heatmap_data:
        heatmap_array = np.log10(np.array(heatmap_data) + 1e-16)
        im = plt.imshow(heatmap_array, aspect='auto', cmap='viridis_r')
        plt.colorbar(im, label='log10(loss)')
        plt.yticks(range(len(layer_list)), [f'{layer_num}L' for layer_num in layer_list])
        plt.xticks(range(len(fixed_list)), [f'{k}' for k in fixed_list], rotation=90, fontsize=8)
        plt.xlabel('Fixed Parameter k', fontsize=12)
        plt.ylabel('Number of Hidden Layers', fontsize=12)
        plt.title('Loss Heatmap: Fixed Parameter k vs Number of Hidden Layers\n(RELU, 12000 points, 100 reps, No early stopping)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/loss_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Heatmap for time
    plt.figure(figsize=(12, 8))
    heatmap_data_time = []
    for layer_num in layer_list:
        row_data = []
        for k in fixed_list:
            if k in all_results[layer_num] and 'mean_duration' in all_results[layer_num][k]:
                row_data.append(all_results[layer_num][k]['mean_duration'])
        if row_data:
            heatmap_data_time.append(row_data)
    if heatmap_data_time:
        im = plt.imshow(heatmap_data_time, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='Training Time (s)')
        plt.yticks(range(len(layer_list)), [f'{layer_num}L' for layer_num in layer_list])
        plt.xticks(range(len(fixed_list)), [f'{k}' for k in fixed_list], rotation=90, fontsize=8)
        plt.xlabel('Fixed Parameter k', fontsize=12)
        plt.ylabel('Number of Hidden Layers', fontsize=12)
        plt.title('Training Time Heatmap: Fixed Parameter k vs Number of Hidden Layers\n(RELU, 12000 points, 100 reps, No early stopping)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/time_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"Plots saved to: {save_dir}")

# ============================================================
# 13. Main Function
# ============================================================
def main():
    print("=" * 120)
    print("Hybrid Network Experiment with Variable Layers - 1 A100 GPU + 16 CPU Cores")
    print("Network Design: Fixed and trainable parts receive entire output from previous layer")
    print("Activation Function: RELU")
    print("New target function: sin(2x) + cos(5x) + 0.3*sin(8x) + 0.2*cos(11x) + 0.1*sin(13x)")
    print("12000 data points, 100 repetitions, NO EARLY STOPPING")
    print("=" * 120)
    
    all_results, layer_list, fixed_list, save_dir, total_time = run_single_gpu_experiment()
    if all_results is None:
        print("Training failed!")
        return
    
    save_results_to_txt(all_results, layer_list, fixed_list, save_dir, total_time)
    generate_plots(all_results, layer_list, fixed_list, save_dir)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {save_dir}")
    return all_results, save_dir

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    all_results, save_dir = main()