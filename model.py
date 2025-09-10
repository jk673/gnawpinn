import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Optional, List

class MeshGraphNetsProcessor(MessagePassing):
    """
    Fixed MeshGraphNets-style processor 
    """
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
            
            # Message passing: aggregate edge attributes to nodes
            x_residual = x
            
            # Manual aggregation (more reliable)
            num_nodes = x.size(0)
            x_aggregated = torch.zeros_like(x)
            
            # Add edge attributes to destination nodes
            x_aggregated.index_add_(0, col, edge_attr)
            
            # Count neighbors for averaging
            ones = torch.ones(edge_index.size(1), 1, device=x.device, dtype=x.dtype)
            count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
            count.index_add_(0, col, ones)
            count = count.clamp(min=1)  # Avoid division by zero
            
            # Average aggregation
            x_aggregated = x_aggregated / count
            
            # Update nodes
            node_input = torch.cat([x_residual, x_aggregated], dim=1)
            x_update = self.node_models[i](node_input)
            x = x_residual + x_update
            
        return x, edge_attr
    
    def message(self, x_j, edge_attr):
        """This won't be used since we do manual aggregation"""
        return edge_attr
    
    def update(self, aggr_out):
        """This won't be used since we do manual aggregation"""
        return aggr_out

class SimpleMeshProcessor(nn.Module):
    """
    Alternative simpler mesh processor without MessagePassing inheritance
    More compatible across different PyTorch Geometric versions
    """
    def __init__(self, latent_size=128, num_layers=15, dropout=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.num_layers = num_layers
        
        # Edge and node MLPs
        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()
        
        for _ in range(num_layers):
            self.edge_models.append(nn.Sequential(
                nn.Linear(latent_size * 3, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))
            
            self.node_models.append(nn.Sequential(
                nn.Linear(latent_size * 2, latent_size * 2),
                nn.LayerNorm(latent_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_size * 2, latent_size),
                nn.LayerNorm(latent_size)
            ))
    
    def aggregate_neighbors(self, x, edge_index, edge_attr):
        """Manual aggregation of neighbor features"""
        row, col = edge_index
        
        # Create aggregation matrix
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        # Aggregate edge attributes to destination nodes
        aggregated.index_add_(0, col, edge_attr)
        
        # Count neighbors for mean aggregation
        ones = torch.ones(edge_index.size(1), 1, device=x.device)
        count = torch.zeros(num_nodes, 1, device=x.device)
        count.index_add_(0, col, ones)
        count = count.clamp(min=1)  # Avoid division by zero
        
        # Mean aggregation
        aggregated = aggregated / count
        
        return aggregated
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        for i in range(self.num_layers):
            # Update edges
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_attr = edge_attr + self.edge_models[i](edge_input)
            
            # Aggregate neighbor information
            x_residual = x
            x_aggregated = self.aggregate_neighbors(x, edge_index, edge_attr)
            
            # Update nodes
            x = x_residual + self.node_models[i](torch.cat([x_residual, x_aggregated], dim=1))
            
        return x, edge_attr

class CFDSurrogateModel(torch.nn.Module):
    """Complete CFD Surrogate Model with MeshGraphNets processor"""
    def __init__(self, node_feat_dim=7, hidden_dim=128, output_dim=4, num_mp_layers=10, 
                 edge_feat_dim=8):
        super().__init__()
        
        self.node_feat_dim = node_feat_dim  # Default: [x, y, z, normal_x, normal_y, normal_z, area]
        self.hidden_dim = hidden_dim
        
        # Input encoding for 7D node features
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU()
        )
        
        # Edge feature encoder
        self.edge_encoder = torch.nn.Linear(edge_feat_dim, hidden_dim)
        
        # Always use MeshGraphNets processor (fixed compatibility)
        self.processor = MeshGraphNetsProcessor(
            latent_size=hidden_dim,
            num_layers=num_mp_layers,
            dropout=0.1
        )
        
        # Output decoding
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        # Ensure node features exist (expect 7D: [x, y, z, normal_x, normal_y, normal_z, area])
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("Node features (data.x) must be provided with 7 dimensions: [x, y, z, normal_x, normal_y, normal_z, area]")
        
        # Validate node feature dimensions
        if data.x.shape[1] != self.node_feat_dim:
            raise ValueError(f"Expected {self.node_feat_dim}D node features, got {data.x.shape[1]}D. "
                           "Required: [x, y, z, normal_x, normal_y, normal_z, area]")
        
        # Ensure pos field exists for physics loss calculations
        if not hasattr(data, 'pos') or data.pos is None:
            # Create pos from first 3 dimensions of x (coordinates)
            data.pos = data.x[:, :3].clone().requires_grad_(True)
        else:
            # Ensure pos requires gradient for physics loss
            if not data.pos.requires_grad:
                data.pos = data.pos.requires_grad_(True)
        
        # Encode node features (7D → hidden_dim)
        x = self.encoder(data.x)
        
        # Process edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.shape[1] != self.edge_encoder.in_features:
                raise ValueError(f"Expected {self.edge_encoder.in_features}D edge features, got {data.edge_attr.shape[1]}D")
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            # Create edge features if not provided
            edge_attr = self._create_edge_features(data, x.device)
        
        # Get batch if available
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Message passing with MeshGraphNets
        x, edge_attr = self.processor(x, data.edge_index, edge_attr, batch)
        
        # Decode to output (hidden_dim → 4D: [pressure_coeff, tau_x, tau_y, tau_z])
        out = self.decoder(x)
        
        return out
    
    def _create_edge_features(self, data, device):
        """Create edge features from node geometry if not provided"""
        row, col = data.edge_index
        
        # Extract positions (first 3 dimensions)
        pos_i = data.x[row, :3]  # Source positions
        pos_j = data.x[col, :3]  # Target positions
        
        # Extract normals (dimensions 3-5)
        normal_i = data.x[row, 3:6]
        normal_j = data.x[col, 3:6]
        
        # Extract areas (dimension 6)
        area_i = data.x[row, 6:7]
        area_j = data.x[col, 6:7]
        
        # Compute edge features (8D)
        edge_vec = pos_j - pos_i                    # [3D] Edge vector
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)  # [1D] Edge length
        
        # Dot product of normals (geometric relationship)
        normal_dot = torch.sum(normal_i * normal_j, dim=1, keepdim=True)  # [1D]
        
        # Area ratio
        area_ratio = area_i / (area_j + 1e-8)       # [1D] Area ratio
        
        # Combined geometric features
        edge_dir = edge_vec / (edge_length + 1e-8)  # [3D] Normalized edge direction
        
        # Concatenate all edge features [3 + 1 + 1 + 1 + 2] = 8D
        edge_attr = torch.cat([
            edge_dir,        # [3D] Edge direction
            edge_length,     # [1D] Edge length  
            normal_dot,      # [1D] Normal alignment
            area_ratio,      # [1D] Area ratio
            area_i,          # [1D] Source area
            area_j           # [1D] Target area
        ], dim=1)
        
        return self.edge_encoder(edge_attr)


class GeometricMultiGridEncoder(nn.Module):
    """
    Multi-grid approach for handling different mesh resolutions.
    Fixed version that properly handles different node counts.
    """
    def __init__(self, input_dim, hidden_dim, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        
        # Encoders for each level
        self.level_encoders = nn.ModuleList()
        for level in range(num_levels):
            self.level_encoders.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Pooling to aggregate node features to graph level
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        
        # Fusion layer combines aggregated features from all levels
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, features_list: List[torch.Tensor]):
        """
        features_list: List of features at different resolutions
        Each tensor has shape (num_nodes, input_dim) where num_nodes can differ
        Returns: Single fused representation of shape (hidden_dim,)
        """
        encoded = []
        
        # Encode and pool each level
        for level, features in enumerate(features_list):
            if level < len(self.level_encoders):
                # Encode features
                level_encoded = self.level_encoders[level](features)
                
                # Pool to graph level (mean pooling over nodes)
                pooled = level_encoded.mean(dim=0, keepdim=True)  # Shape: (1, hidden_dim)
                pooled = self.pool(pooled)
                
                encoded.append(pooled)
        
        # Concatenate and fuse
        if encoded:
            # All pooled features have shape (1, hidden_dim), so concatenation is valid
            fused = torch.cat(encoded, dim=-1)  # Shape: (1, hidden_dim * num_levels)
            return self.fusion(fused)
        else:
            return torch.zeros(1, self.hidden_dim)

class MultiResolutionProcessor(nn.Module):
    """
    Process multiple resolution levels with proper interpolation
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # Encoders for different resolutions
        self.coarse_encoder = nn.Linear(input_dim, hidden_dim)
        self.fine_encoder = nn.Linear(input_dim, hidden_dim)
        
        # Interpolation network
        self.interpolator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def interpolate_features(self, coarse_features, fine_shape):
        """
        Interpolate coarse features to match fine resolution
        Simple nearest neighbor interpolation for demonstration
        """
        # This is a simplified version - in practice you'd use mesh connectivity
        coarse_size = coarse_features.size(0)
        fine_size = fine_shape
        
        # Repeat coarse features to approximate fine resolution
        repeat_factor = (fine_size + coarse_size - 1) // coarse_size
        interpolated = coarse_features.repeat(repeat_factor, 1)[:fine_size]
        
        return interpolated
    
    def forward(self, coarse_data, fine_data):
        """Process multi-resolution data"""
        # Encode both resolutions
        coarse_encoded = self.coarse_encoder(coarse_data)
        fine_encoded = self.fine_encoder(fine_data)
        
        # Interpolate coarse to fine resolution
        coarse_interpolated = self.interpolate_features(
            coarse_encoded, fine_data.size(0)
        )
        
        # Combine features
        combined = torch.cat([fine_encoded, coarse_interpolated], dim=-1)
        fused = self.interpolator(combined)
        
        return fused

class EnsemblePredictor(nn.Module):
    """Ensemble of models for uncertainty quantification"""
    def __init__(self, base_model_class, num_models=5, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            base_model_class(**model_kwargs) for _ in range(num_models)
        ])
    
    def forward(self, data, return_all=False):
        predictions = []
        for model in self.models:
            pred = model(data)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        if return_all:
            return predictions
        else:
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            return mean, std

class LossBalancer:
    """Dynamic loss balancing for multi-task learning"""
    def __init__(self, num_losses=3, alpha=0.9):
        self.num_losses = num_losses
        self.alpha = alpha
        self.loss_weights = torch.ones(num_losses)
        self.prev_losses = None
    
    def update_weights(self, current_losses):
        if self.prev_losses is not None:
            loss_ratios = []
            for i in range(self.num_losses):
                if self.prev_losses[i] > 0:
                    ratio = current_losses[i] / self.prev_losses[i]
                    loss_ratios.append(ratio)
                else:
                    loss_ratios.append(1.0)
            
            loss_ratios = torch.tensor(loss_ratios)
            self.loss_weights = (self.alpha * self.loss_weights + 
                                (1 - self.alpha) * loss_ratios)
            self.loss_weights = self.loss_weights / self.loss_weights.sum() * self.num_losses
        
        self.prev_losses = current_losses.clone()
        return self.loss_weights

class DataAugmentation:
    """Data augmentation strategies for CFD meshes"""
    
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """Add Gaussian noise to node features"""
        data.x = data.x + torch.randn_like(data.x) * noise_level
        if hasattr(data, 'pos'):
            data.pos = data.pos + torch.randn_like(data.pos) * noise_level * 0.1
        return data
    
    @staticmethod
    def random_rotation(data):
        """Random rotation augmentation (for 3D data)"""
        if not hasattr(data, 'pos'):
            return data
            
        angle = torch.rand(1) * 2 * np.pi
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        
        # 2D rotation around z-axis
        rot_matrix = torch.tensor([
            [cos_a.item(), -sin_a.item(), 0],
            [sin_a.item(), cos_a.item(), 0],
            [0, 0, 1]
        ], dtype=data.pos.dtype)
        
        data.pos = torch.matmul(data.pos, rot_matrix.T)
        return data

class AdaptiveSampling(nn.Module):
    """Adaptive sampling strategy for important regions"""
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, k=10):
        """Select k most important nodes"""
        importance = self.importance_net(x).squeeze(-1)
        _, top_indices = torch.topk(importance, min(k, x.size(0)))
        
        mask = torch.zeros(x.size(0), dtype=torch.bool)
        mask[top_indices] = True
        
        return mask, importance