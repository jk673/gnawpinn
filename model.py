import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import Optional, List

class MeshGraphNetsProcessor(MessagePassing):
    """
    Enhanced MeshGraphNets-style processor with improvements
    Compatible with various PyTorch Geometric versions
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
        
        # Store edge attributes as class variable for message passing
        self.edge_attr = None
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        for i in range(self.num_layers):
            # Update edges
            row, col = edge_index
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_attr = edge_attr + self.edge_models[i](edge_input)
            
            # Store edge_attr for message function
            self.edge_attr = edge_attr
            
            # Update nodes with message passing
            x_residual = x
            x_aggregated = self.propagate(edge_index, size=(x.size(0), x.size(0)))
            x = x_residual + self.node_models[i](torch.cat([x_residual, x_aggregated], dim=1))
            
        return x, edge_attr
    
    def message(self, x_j=None):
        """Define how messages are computed"""
        # Use stored edge attributes as messages
        return self.edge_attr
    
    def update(self, aggr_out):
        """Define how node embeddings are updated"""
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
    """Complete CFD Surrogate Model with flexible processor choice"""
    def __init__(self, node_feat_dim, hidden_dim, output_dim, num_mp_layers=10, 
                 edge_feat_dim=8, use_simple=True):  # Added edge_feat_dim parameter
        super().__init__()
        
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        
        # Input encoding - will be created dynamically if needed
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU()
        )
        
        # Edge feature encoder - now accepts edge_feat_dim dimensions
        self.edge_encoder = torch.nn.Linear(edge_feat_dim, hidden_dim)
        
        # Choose processor based on compatibility
        if use_simple:
            self.processor = SimpleMeshProcessor(
                latent_size=hidden_dim,
                num_layers=num_mp_layers,
                dropout=0.1
            )
        else:
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
        # Ensure node features exist
        if not hasattr(data, 'x') or data.x is None:
            # Fallback: create node features from positions if available
            if hasattr(data, 'pos') and data.pos is not None:
                data.x = data.pos
            else:
                # Last resort: create dummy features
                num_nodes = data.edge_index.max().item() + 1
                data.x = torch.randn(num_nodes, 3, device=data.edge_index.device)
        
        # Handle dimension mismatch by recreating encoder if needed
        actual_feat_dim = data.x.shape[1]
        if actual_feat_dim != self.node_feat_dim:
            # Recreate encoder with correct input dimension
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(actual_feat_dim, self.hidden_dim),
                torch.nn.LayerNorm(self.hidden_dim),
                torch.nn.GELU()
            ).to(data.x.device)
            self.node_feat_dim = actual_feat_dim
        
        # Encode node features
        x = self.encoder(data.x)
        
        # Process edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
            # No need to unsqueeze since edge_attr is already [E, 8]
            # Transform to hidden_dim
            edge_attr = self.edge_encoder(edge_attr)
        else:
            # Create edge features based on distance if not provided
            row, col = data.edge_index
            if hasattr(data, 'pos'):
                edge_vec = data.pos[col] - data.pos[row]
                edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
                # Pad to match edge_feat_dim if creating from scratch
                edge_attr = torch.cat([edge_length] + [torch.zeros_like(edge_length)] * 7, dim=1)
                edge_attr = self.edge_encoder(edge_attr)
            else:
                # Random initialization as fallback
                edge_attr = torch.randn(data.edge_index.size(1), x.size(1), device=x.device)
        
        # Get batch if available
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Message passing
        x, edge_attr = self.processor(x, data.edge_index, edge_attr, batch)
        
        # Decode to output
        out = self.decoder(x)
        
        return out

class VortexNetCorrection(nn.Module):
    """VortexNet-style multi-fidelity correction network"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        self.correction_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, low_fidelity, features):
        combined = torch.cat([low_fidelity, features], dim=-1)
        correction = self.correction_net(combined)
        confidence = self.confidence_net(features)
        high_fidelity = low_fidelity + confidence * correction
        return high_fidelity, confidence

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