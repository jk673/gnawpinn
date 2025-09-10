"""
GNAWPINN Unified Framework
Complete physics-informed neural network for CFD applications

Combines:
- CFD Surrogate Model (7D MeshGraphNets)
- Comprehensive Physics Loss
- Enhanced Training with Component Metrics
- Validation with Detailed Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import wandb
import gc
import psutil
import os

# ============================
# MODEL ARCHITECTURE
# ============================

class MeshGraphNetsProcessor(MessagePassing):
    """Fixed MeshGraphNets processor with manual aggregation"""
    
    def __init__(self, latent_size=128, num_layers=15, dropout=0.1):
        super().__init__(aggr='add')
        self.latent_size = latent_size
        self.num_layers = num_layers
        
        # Edge and node MLPs for each layer
        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()
        
        for _ in range(num_layers):
            # Edge model
            self.edge_models.append(nn.Sequential(
                nn.Linear(latent_size * 3, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))
            
            # Node model  
            self.node_models.append(nn.Sequential(
                nn.Linear(latent_size * 2, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        for i in range(self.num_layers):
            # Update edges
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_attr_new = self.edge_models[i](edge_input)
            edge_attr = edge_attr + edge_attr_new
            
            # Manual aggregation
            x_residual = x
            num_nodes = x.size(0)
            x_aggregated = torch.zeros_like(x)
            
            # Add edge attributes to destination nodes
            x_aggregated.index_add_(0, col, edge_attr)
            
            # Count neighbors for averaging
            ones = torch.ones(edge_index.size(1), 1, device=x.device, dtype=x.dtype)
            count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            count.index_add_(0, col, ones)
            count = count.clamp(min=1)
            
            # Average aggregation
            x_aggregated = x_aggregated / count
            
            # Update nodes
            node_input = torch.cat([x_residual, x_aggregated], dim=1)
            x_update = self.node_models[i](node_input)
            x = x_residual + x_update
            
        return x, edge_attr


class CFDSurrogateModel(nn.Module):
    """Complete CFD Surrogate Model with 7D MeshGraphNets"""
    
    def __init__(self, node_feat_dim=7, edge_feat_dim=8, hidden_dim=128, 
                 output_dim=4, num_mp_layers=10):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        
        # Input encoding for 7D node features
        self.encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        
        # MeshGraphNets processor
        self.processor = MeshGraphNetsProcessor(
            latent_size=hidden_dim,
            num_layers=num_mp_layers,
            dropout=0.1
        )
        
        # Output decoding
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        # Validate input
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("Node features (data.x) must be provided")
        
        if data.x.shape[1] != self.node_feat_dim:
            raise ValueError(f"Expected {self.node_feat_dim}D node features, got {data.x.shape[1]}D")
        
        # Ensure pos field exists for physics loss
        if not hasattr(data, 'pos') or data.pos is None:
            data.pos = data.x[:, :3].clone().requires_grad_(True)
        else:
            if not data.pos.requires_grad:
                data.pos = data.pos.requires_grad_(True)
        
        # Encode node features
        x = self.encoder(data.x)
        
        # Process edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = self._create_edge_features(data, x.device)
        
        # Message passing
        x, edge_attr = self.processor(x, data.edge_index, edge_attr)
        
        # Decode to output
        out = self.decoder(x)
        
        return out
    
    def _create_edge_features(self, data, device):
        """Create 8D edge features from geometry"""
        row, col = data.edge_index
        
        # Extract positions and properties
        pos_i, pos_j = data.x[row, :3], data.x[col, :3]
        normal_i, normal_j = data.x[row, 3:6], data.x[col, 3:6]
        area_i, area_j = data.x[row, 6:7], data.x[col, 6:7]
        
        # Compute edge features
        edge_vec = pos_j - pos_i
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
        edge_dir = edge_vec / (edge_length + 1e-8)
        normal_dot = torch.sum(normal_i * normal_j, dim=1, keepdim=True)
        area_ratio = area_i / (area_j + 1e-8)
        
        # Concatenate 8D features
        edge_attr = torch.cat([
            edge_dir,        # [3D] Edge direction
            edge_length,     # [1D] Edge length  
            normal_dot,      # [1D] Normal alignment
            area_ratio,      # [1D] Area ratio
            area_i,          # [1D] Source area
            area_j           # [1D] Target area
        ], dim=1)
        
        return self.edge_encoder(edge_attr)


# ============================
# COMPREHENSIVE PHYSICS LOSS
# ============================

class ComprehensivePhysicsLoss(nn.Module):
    """Comprehensive physics loss with guaranteed active penalties"""
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.default_weights = {
            'mse': 1.0,
            'pressure_range_penalty': 0.5,
            'shear_range_penalty': 0.4, 
            'component_balance_penalty': 0.3,
            'physical_consistency_loss': 0.4,
            'spatial_coherence_loss': 0.2,
            'shear_stress_smoothness': 0.25
        }
        
        self.loss_weights = loss_weights or self.default_weights.copy()
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    data: Any) -> Dict[str, torch.Tensor]:
        """Compute comprehensive physics loss"""
        device = predictions.device
        loss_components = {}
        
        # 1. Basic MSE
        mse_loss = F.mse_loss(predictions, targets)
        loss_components['mse'] = mse_loss
        
        # 2. Pressure range penalty
        p_coeff = predictions[:, 0]
        abs_p = torch.abs(p_coeff)
        pressure_penalty = F.relu(abs_p - 2.0).mean() + 1e-6 * abs_p.mean()
        extreme_pressure = F.relu(abs_p - 5.0).mean()
        loss_components['pressure_range_penalty'] = pressure_penalty + 10.0 * extreme_pressure
        
        # 3. Shear stress penalties
        tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
        tau_magnitude = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-8)
        
        # Magnitude penalty
        shear_penalty = F.relu(tau_magnitude - 1.0).mean() + 1e-6 * tau_magnitude.mean()
        extreme_shear = F.relu(tau_magnitude - 5.0).mean()
        loss_components['shear_range_penalty'] = shear_penalty + 10.0 * extreme_shear
        
        # Component balance penalty
        tau_components = torch.stack([torch.abs(tau_x), torch.abs(tau_y), torch.abs(tau_z)], dim=1)
        max_component = torch.max(tau_components, dim=1)[0]
        mean_component = torch.mean(tau_components, dim=1)
        balance_penalty = F.relu(max_component - 2.0 * mean_component).mean()
        loss_components['component_balance_penalty'] = balance_penalty + 1e-6 * max_component.mean()
        
        # 4. Physical consistency
        if len(p_coeff) > 2:
            p_centered = p_coeff - torch.mean(p_coeff)
            tau_centered = tau_magnitude - torch.mean(tau_magnitude)
            
            numerator = torch.sum(p_centered * tau_centered)
            p_var = torch.sum(p_centered**2)
            tau_var = torch.sum(tau_centered**2)
            
            if p_var > 1e-8 and tau_var > 1e-8:
                correlation = numerator / torch.sqrt(p_var * tau_var + 1e-8)
                correlation_loss = (1.0 - torch.abs(correlation))**2
            else:
                correlation_loss = torch.tensor(1.0, device=device)
        else:
            correlation_loss = torch.tensor(1.0, device=device)
            
        loss_components['physical_consistency_loss'] = correlation_loss + 1e-5 * (torch.mean(torch.abs(p_coeff)) + torch.mean(tau_magnitude))
        
        # 5. Spatial coherence
        if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
            row, col = data.edge_index
            edge_diff = predictions[row] - predictions[col]
            coherence_loss = torch.mean(edge_diff**2)
        else:
            coherence_loss = torch.tensor(1e-4, device=device)
            
        loss_components['spatial_coherence_loss'] = coherence_loss + 1e-6 * torch.mean(predictions**2)
        
        # 6. Smoothness
        smoothness_loss = torch.tensor(0.0, device=device)
        if len(predictions) > 1:
            for i in range(4):
                diff = torch.diff(predictions[:, i])
                smoothness_loss += torch.mean(diff**2)
            smoothness_loss /= 4
        loss_components['shear_stress_smoothness'] = smoothness_loss + 1e-6 * torch.mean(predictions**2)
        
        # 7. Compute weighted total
        total_loss = torch.tensor(0.0, device=device)
        # FIXED: Create static copy to avoid dictionary modification during iteration
        loss_items = [(k, v) for k, v in loss_components.items()]
        for loss_name, loss_value in loss_items:
            if isinstance(loss_value, torch.Tensor) and loss_value.dim() == 0:
                weight = self.loss_weights.get(loss_name, 0.0)
                if weight > 0:
                    weighted_loss = weight * loss_value
                    total_loss = total_loss + weighted_loss
                    loss_components[f'{loss_name}_weighted'] = weighted_loss
        
        loss_components['total_loss'] = total_loss
        
        return loss_components


# ============================
# COMPONENT-WISE METRICS
# ============================

def compute_component_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute detailed metrics for each output component"""
    component_names = ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']
    metrics = {}
    
    for i, comp_name in enumerate(component_names):
        pred_comp = predictions[:, i]
        target_comp = targets[:, i]
        
        # Relative L2 error
        error_l2 = torch.norm(pred_comp - target_comp, p=2)
        target_l2 = torch.norm(target_comp, p=2)
        relative_l2 = error_l2 / (target_l2 + 1e-10)
        
        # Additional metrics
        mse_comp = F.mse_loss(pred_comp, target_comp)
        mae_comp = F.l1_loss(pred_comp, target_comp)
        max_abs_error = torch.max(torch.abs(pred_comp - target_comp))
        
        # R² score
        target_mean = torch.mean(target_comp)
        ss_tot = torch.sum((target_comp - target_mean) ** 2)
        ss_res = torch.sum((target_comp - pred_comp) ** 2)
        r2_score = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Store metrics
        metrics[f'{comp_name}_relative_l2'] = relative_l2
        metrics[f'{comp_name}_mse'] = mse_comp
        metrics[f'{comp_name}_mae'] = mae_comp
        metrics[f'{comp_name}_max_error'] = max_abs_error
        metrics[f'{comp_name}_r2'] = r2_score
    
    # Overall metrics
    overall_l2 = torch.norm(predictions - targets, p=2) / torch.norm(targets, p=2)
    metrics['overall_relative_l2'] = overall_l2
    metrics['overall_mse'] = F.mse_loss(predictions, targets)
    
    return metrics


# ============================
# ENHANCED TRAINING FUNCTIONS
# ============================

def enhanced_train_epoch(model, train_loader, optimizer, scheduler, epoch, config, device):
    """Enhanced training epoch with comprehensive physics loss and metrics"""
    
    physics_loss_fn = ComprehensivePhysicsLoss(config.get('loss_weights', None))
    physics_loss_fn = physics_loss_fn.to(device)
    
    model.train()
    total_loss = 0
    loss_components_sum = {}
    
    # For component analysis
    train_predictions = []
    train_targets = []
    
    num_batches = len(train_loader)
    successful_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            if not predictions.requires_grad:
                predictions = predictions.requires_grad_(True)
            
            # Store for component analysis (periodic)
            if batch_idx % 10 == 0:
                train_predictions.append(predictions.detach())
                train_targets.append(batch.y)
            
            # Compute comprehensive physics loss
            loss_result = physics_loss_fn.compute_loss(predictions, batch.y, batch)
            loss = loss_result['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Track losses
            total_loss += loss.item()
            successful_batches += 1
            
            # Accumulate loss components
            loss_items = [(k, v) for k, v in loss_result.items() 
                         if isinstance(v, torch.Tensor) and v.dim() == 0]
            
            for key, value in loss_items:
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0
                loss_components_sum[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'mse': loss_result.get('mse', torch.tensor(0)).item(),
                'pressure_penalty': loss_result.get('pressure_range_penalty', torch.tensor(0)).item(),
                'shear_penalty': loss_result.get('shear_range_penalty', torch.tensor(0)).item()
            })
                
        except Exception as e:
            print(f"\\nError in batch {batch_idx}: {e}")
            continue
    
    # Step scheduler
    if scheduler and hasattr(scheduler, 'step') and not hasattr(scheduler, 'mode'):
        scheduler.step()
    
    # Average losses
    avg_loss = total_loss / max(1, successful_batches)
    avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    # Component analysis (periodic)
    if train_predictions and epoch % 5 == 0:
        all_train_pred = torch.cat(train_predictions, dim=0)
        all_train_targ = torch.cat(train_targets, dim=0)
        train_metrics = compute_component_metrics(all_train_pred, all_train_targ)
        
        # Log to WandB
        try:
            train_wandb_log = {'train/loss': avg_loss, 'epoch': epoch}
            for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
                train_wandb_log[f'train/errors/{comp}_relative_l2'] = train_metrics[f'{comp}_relative_l2'].item()
                train_wandb_log[f'train/errors/{comp}_r2'] = train_metrics[f'{comp}_r2'].item()
            
            # Add physics penalty terms
            for key in ['pressure_range_penalty', 'shear_range_penalty', 'component_balance_penalty', 'physical_consistency_loss']:
                if key in avg_components:
                    train_wandb_log[f'train/physics/{key}'] = avg_components[key]
            
            wandb.log(train_wandb_log)
        except:
            pass
    
    return avg_loss, avg_components


def enhanced_validate_epoch(model, val_loader, epoch, config, device):
    """Enhanced validation with detailed component analysis"""
    
    physics_loss_fn = ComprehensivePhysicsLoss(config.get('loss_weights', None))
    physics_loss_fn = physics_loss_fn.to(device)
    
    model.eval()
    total_loss = 0
    loss_components_sum = {}
    
    # For component analysis
    all_predictions = []
    all_targets = []
    
    num_batches = len(val_loader)
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                batch = batch.to(device)
                predictions = model(batch)
                
                # Accumulate for analysis
                all_predictions.append(predictions)
                all_targets.append(batch.y)
                
                # Compute loss
                loss_result = physics_loss_fn.compute_loss(predictions, batch.y, batch)
                loss = loss_result['total_loss']
                total_loss += loss.item()
                successful_batches += 1
                
                # Accumulate components
                loss_items = [(k, v) for k, v in loss_result.items() 
                             if isinstance(v, torch.Tensor) and v.dim() == 0]
                
                for key, value in loss_items:
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0
                    loss_components_sum[key] += value.item()
                        
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    # Average losses
    avg_loss = total_loss / max(1, successful_batches)
    avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    # Component analysis
    if all_predictions:
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        component_metrics = compute_component_metrics(all_pred, all_targ)
        
        # Print results
        print(f"\\n📊 Validation Results (Epoch {epoch+1}):")
        print(f"{'Component':<15} {'Rel L2':<10} {'R²':<8} {'MSE':<10}")
        print("-" * 50)
        
        for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
            rel_l2 = component_metrics[f'{comp}_relative_l2'].item()
            r2 = component_metrics[f'{comp}_r2'].item()
            mse = component_metrics[f'{comp}_mse'].item()
            print(f"{comp:<15} {rel_l2:<10.6f} {r2:<8.4f} {mse:<10.6f}")
        
        print(f"\\n🔥 Physics Penalties:")
        for key in ['pressure_range_penalty', 'shear_range_penalty', 'component_balance_penalty']:
            if key in avg_components:
                print(f"  {key}: {avg_components[key]:.6f}")
    
    # Log to WandB
    try:
        wandb_log = {'val/loss': avg_loss, 'epoch': epoch}
        
        # Component metrics
        if all_predictions:
            for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
                wandb_log[f'val/errors/{comp}_relative_l2'] = component_metrics[f'{comp}_relative_l2'].item()
                wandb_log[f'val/errors/{comp}_r2'] = component_metrics[f'{comp}_r2'].item()
        
        # Physics penalties
        for key in ['pressure_range_penalty', 'shear_range_penalty', 'component_balance_penalty', 'physical_consistency_loss']:
            if key in avg_components:
                wandb_log[f'val/physics/{key}'] = avg_components[key]
        
        wandb.log(wandb_log)
    except:
        pass
    
    return avg_loss, avg_components


# ============================
# UTILITY FUNCTIONS
# ============================

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1e9
    gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {'cpu_gb': cpu_memory, 'gpu_gb': gpu_memory}

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_test_data(num_nodes=100, device='cpu'):
    """Create test data for debugging"""
    # 7D node features
    x = torch.randn(num_nodes, 7, device=device)
    x[:, :3] *= 5  # positions
    x[:, 3:6] = F.normalize(x[:, 3:6], dim=1)  # unit normals
    x[:, 6] = torch.abs(x[:, 6]) * 0.1  # positive areas
    
    # Edges
    edge_index = torch.stack([
        torch.arange(num_nodes - 1),
        torch.arange(1, num_nodes)
    ]).to(device)
    
    # 4D targets
    y = torch.randn(num_nodes, 4, device=device) * 0.1
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

print("🚀 GNAWPINN Unified Framework Loaded!")
print("\\n✅ Complete system with:")
print("  - CFD Surrogate Model (7D → 4D)")  
print("  - Comprehensive Physics Loss (GUARANTEED active penalties)")
print("  - Enhanced Training with Component Metrics")
print("  - Detailed Validation Analysis")
print("  - Memory Management")
print("\\n🎯 Ready for production training!")