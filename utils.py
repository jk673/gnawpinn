import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from pathlib import Path
import gc
import warnings
from typing import Union, List, Optional, Any, Dict
import psutil
import time
from tqdm.auto import tqdm

# PyTorch Geometric imports with fallbacks
try:
    from torch_geometric.data import Data
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    Data = None
    PYGEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Some functionality may be limited.")

# Edge features import
try:
    from edge_features import compute_enhanced_edge_features
    EDGE_FEATURES_AVAILABLE = True
except ImportError:
    compute_enhanced_edge_features = None
    EDGE_FEATURES_AVAILABLE = False
    warnings.warn("Edge features module not available.")

# Type definitions
AnyPath = Union[str, bytes, Path]

def create_sample_data(num_nodes=100, num_features=10, num_outputs=2):
    """Create sample CFD mesh data"""
    # Create random edges
    num_edges = num_nodes * 3
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    # Make undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Remove duplicates (simple approach)
    edge_index = torch.unique(edge_index, dim=1)
    
    data = Data(
        x=torch.randn(num_nodes, num_features),
        edge_index=edge_index,
        pos=torch.randn(num_nodes, 3),
        y=torch.randn(num_nodes, num_outputs)
    )
    
    return data

def compute_physics_loss(pred, data):
    """Example physics-based loss"""
    if pred.size(-1) >= 3:
        div = torch.mean(torch.abs(pred[:, 0] - pred[:, 0].mean()))
        return div * 0.1
    return torch.tensor(0.0)

def compute_smoothness_loss(pred, edge_index):
    """Smoothness regularization loss"""
    row, col = edge_index
    edge_diff = pred[row] - pred[col]
    return torch.mean(edge_diff.pow(2))



"""
Unified utility functions for graph processing and data management.

This module provides comprehensive utilities for:
- Graph data preprocessing and validation
- Memory management and performance monitoring
- CFD/Flow-specific graph operations
- Progress tracking and file I/O
- Surface property computation

Key Features:
- Robust error handling and validation
- Memory-efficient operations
- CFD-specialized edge features
- Comprehensive logging and monitoring
"""

import gc
import warnings
from pathlib import Path
from typing import Union, List, Optional, Any, Dict
import psutil
import time

import torch
from tqdm.auto import tqdm

# PyTorch Geometric imports with fallbacks
try:
    from torch_geometric.data import Data
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    Data = None
    PYGEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Some functionality may be limited.")

# Edge features import
try:
    from edge_features import compute_enhanced_edge_features
    EDGE_FEATURES_AVAILABLE = True
except ImportError:
    compute_enhanced_edge_features = None
    EDGE_FEATURES_AVAILABLE = False
    warnings.warn("Edge features module not available.")

# Type definitions
AnyPath = Union[str, bytes, Path]

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_used'] = system_memory.used / 1e9
        memory_info['system_available'] = system_memory.available / 1e9
        memory_info['system_percent'] = system_memory.percent
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            
            memory_info['gpu_total'] = gpu_memory
            memory_info['gpu_allocated'] = allocated
            memory_info['gpu_cached'] = cached
            memory_info['gpu_free'] = gpu_memory - cached
            memory_info['gpu_percent'] = (cached / gpu_memory) * 100
        
        return memory_info
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status and usage delta."""
        current = self._get_memory_usage()
        delta = {}
        
        for key, value in current.items():
            if key in self.start_memory:
                delta[f'{key}_delta'] = value - self.start_memory[key]
        
        return {
            'current': current,
            'delta': delta,
            'elapsed_time': time.time() - self.start_time
        }
    
    def print_status(self, title: str = "Memory Status"):
        """Print formatted memory status."""
        status = self.get_status()
        current = status['current']
        
        print(f"\n🔍 {title}:")
        print(f"  💾 System: {current['system_used']:.2f}GB used "
              f"({current['system_percent']:.1f}%), "
              f"{current['system_available']:.2f}GB available")
        
        if 'gpu_total' in current:
            print(f"  🖥️  GPU: {current['gpu_allocated']:.2f}GB allocated, "
                  f"{current['gpu_cached']:.2f}GB cached "
                  f"({current['gpu_percent']:.1f}%), "
                  f"{current['gpu_free']:.2f}GB free")
        
        if status['delta']:
            delta = status['delta']
            print(f"  📊 Delta: System {delta.get('system_used_delta', 0):.2f}GB")
            if 'gpu_allocated_delta' in delta:
                print(f"         GPU {delta['gpu_allocated_delta']:.2f}GB allocated")


def cleanup_memory(aggressive: bool = False, print_status: bool = True):
    """Comprehensive memory cleanup."""
    if print_status:
        monitor = MemoryMonitor()
        monitor.print_status("Memory Before Cleanup")
    
    collected = gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.ipc_collect()
    
    if aggressive:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if print_status:
        monitor.print_status("Memory After Cleanup")
        print(f"  🗑️  Collected {collected} objects")


def check_gpu_memory():
    """Simple GPU memory check (legacy function)."""
    if torch.cuda.is_available():
        monitor = MemoryMonitor()
        status = monitor.get_status()['current']
        
        print(f"🔍 GPU Memory Status:")
        if 'gpu_total' in status:
            print(f"  Total: {status['gpu_total']:.2f} GB")
            print(f"  Allocated: {status['gpu_allocated']:.2f} GB "
                  f"({status['gpu_percent']:.1f}%)")
            print(f"  Free: {status['gpu_free']:.2f} GB")
            
            if status['gpu_percent'] > 80:
                print("⚠️  High memory usage detected.")
    else:
        print("❌ CUDA not available")

# =============================================================================
# PROGRESS BAR UTILITIES
# =============================================================================

# Default tqdm configuration
_TQDM_KW = {
    'ncols': 100,
    'dynamic_ncols': False,
    'mininterval': 0.25,
    'smoothing': 0.1
}

def create_progress_bar(total: int, desc: str, position: int = 0, 
                       leave: bool = False, **kwargs) -> tqdm:
    """Create a standardized progress bar."""
    default_config = {
        'ncols': 100,
        'dynamic_ncols': False,
        'mininterval': 0.25,
        'smoothing': 0.1,
        'bar_format': "{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt} {postfix}"
    }
    
    config = {**default_config, **kwargs}
    
    return tqdm(
        total=total,
        desc=desc,
        position=position,
        leave=leave,
        **config
    )

# Legacy progress bar functions for compatibility
def _bar(total, desc, position, leave):
    """Create a progress bar with default configuration."""
    return tqdm(total=total, desc=desc, position=position, leave=leave, **_TQDM_KW)

def _mkbar(total, desc, position=0, leave=False, progress=True):
    """Create a configurable progress bar."""
    if not progress:
        return None
    return create_progress_bar(total, desc, position, leave)

def _update(bar, n=1, postfix=None):
    """Update progress bar with optional postfix."""
    if bar is None:
        return
    if postfix is not None:
        if isinstance(postfix, dict):
            bar.set_postfix(postfix)
        else:
            bar.set_postfix_str(str(postfix))
    bar.update(n)

def _close(bar):
    """Close progress bar if it exists."""
    if bar is not None:
        bar.close()

# =============================================================================
# GRAPH PREPROCESSING AND VALIDATION
# =============================================================================

def validate_graph_data(data: Any, required_attrs: Optional[List[str]] = None, 
                       optional_attrs: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate graph data object and report missing attributes."""
    if required_attrs is None:
        required_attrs = ['x', 'y', 'edge_index']
    
    if optional_attrs is None:
        optional_attrs = ['y_graph', 'area', 'normals', 'pos', 'batch', 'edge_attr', 'edge_weight']
    
    results = {
        'is_valid': True,
        'missing_required': [],
        'missing_optional': [],
        'present_attrs': [],
        'attr_shapes': {},
        'attr_dtypes': {},
    }
    
    # Check required attributes
    for attr in required_attrs:
        if hasattr(data, attr):
            value = getattr(data, attr)
            results['present_attrs'].append(attr)
            if hasattr(value, 'shape'):
                results['attr_shapes'][attr] = tuple(value.shape)
            if hasattr(value, 'dtype'):
                results['attr_dtypes'][attr] = str(value.dtype)
        else:
            results['missing_required'].append(attr)
            results['is_valid'] = False
    
    # Check optional attributes
    for attr in optional_attrs:
        if hasattr(data, attr):
            value = getattr(data, attr)
            results['present_attrs'].append(attr)
            if hasattr(value, 'shape'):
                results['attr_shapes'][attr] = tuple(value.shape)
            if hasattr(value, 'dtype'):
                results['attr_dtypes'][attr] = str(value.dtype)
        else:
            results['missing_optional'].append(attr)
    
    return results

def preprocess_graph(data: Any, compute_edge_features: bool = True, 
                    compute_edge_weights: bool = True, eps: float = 1e-8) -> Any:
    """Comprehensive graph preprocessing."""
    # Compute edge features if requested and available
    if compute_edge_features and EDGE_FEATURES_AVAILABLE:
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            try:
                data.edge_attr = compute_enhanced_edge_features(data)
            except Exception as e:
                warnings.warn(f"Failed to compute edge features: {e}")
    
    # Compute edge weights from coordinates
    if compute_edge_weights and not hasattr(data, 'edge_weight'):
        try:
            data.edge_weight = edge_weights_from_coords(data, eps=eps)
        except Exception as e:
            warnings.warn(f"Failed to compute edge weights: {e}")
    
    # Fix y_graph shape if needed
    data = fix_y_graph_shape(data)
    
    return data

def fix_y_graph_shape(data: Any) -> Any:
    """Fix y_graph tensor shape for consistent batch processing."""
    if hasattr(data, "y_graph") and isinstance(data.y_graph, torch.Tensor):
        yg = data.y_graph
        if yg.dim() == 1 and yg.numel() == 2:
            data.y_graph = yg.view(1, 2)  # (2,) -> (1, 2)
        elif yg.dim() == 1 and yg.numel() % 2 == 0:
            data.y_graph = yg.view(-1, 2)  # (2B,) -> (B, 2)
    return data

def edge_weights_from_coords(data: Any, eps: float = 1e-9, squared: bool = False, 
                           clamp_max: Optional[float] = None, clamp: Optional[float] = None) -> torch.Tensor:
    """Compute edge weights based on geometric distances."""
    if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
        raise ValueError("Data must have 'x' and 'edge_index' attributes")
    
    src, dst = data.edge_index
    
    # Extract coordinates (assume first 3 columns are positions)
    coords = data.x[:, :3] if data.x.shape[1] >= 3 else data.x
    diff = coords[dst] - coords[src]
    dist = torch.linalg.norm(diff, dim=1) + eps
    
    # Compute weights
    if squared:
        weights = 1.0 / (dist * dist)
    else:
        weights = 1.0 / dist
    
    # Clamp if requested (support both parameter names)
    clamp_value = clamp_max or clamp
    if clamp_value is not None:
        weights = torch.clamp(weights, max=clamp_value)
    
    return weights.to(torch.float32)

# =============================================================================
# CFD/FLOW-SPECIFIC FEATURES
# =============================================================================

def compute_edge_flow_features(data, include_pressure_gradient=True, include_shear_gradient=True):
    """Compute flow-specific edge features for CFD applications."""
    edge_index = data.edge_index
    src, dst = edge_index
    edge_features = []
    
    # Basic geometric features
    pos = data.x[:, :3]
    diff = pos[dst] - pos[src]
    dist = torch.norm(diff, dim=1, keepdim=True)
    direction = diff / (dist + 1e-8)
    
    edge_features.extend([dist, direction])
    
    # Pressure gradient along edges
    if include_pressure_gradient and data.y.shape[1] >= 1:
        pressure = data.y[:, 0:1]
        pressure_diff = pressure[dst] - pressure[src]
        pressure_gradient = pressure_diff / (dist + 1e-8)
        edge_features.append(pressure_gradient)
    
    # Shear stress gradient
    if include_shear_gradient and data.y.shape[1] >= 4:
        wss = data.y[:, 1:4]
        wss_src = wss[src]
        wss_dst = wss[dst]
        
        wss_mag_src = torch.norm(wss_src, dim=1, keepdim=True)
        wss_mag_dst = torch.norm(wss_dst, dim=1, keepdim=True)
        wss_mag_gradient = (wss_mag_dst - wss_mag_src) / (dist + 1e-8)
        edge_features.append(wss_mag_gradient)
        
        wss_dot = torch.sum(wss_src * wss_dst, dim=1, keepdim=True)
        wss_alignment = wss_dot / (wss_mag_src * wss_mag_dst + 1e-8)
        edge_features.append(wss_alignment)
    
    # Surface curvature features
    if hasattr(data, 'normals'):
        normals = data.normals
        normal_src = normals[src]
        normal_dst = normals[dst]
        
        normal_dot = torch.sum(normal_src * normal_dst, dim=1, keepdim=True)
        normal_angle = torch.acos(torch.clamp(normal_dot, -1+1e-6, 1-1e-6))
        edge_features.append(normal_angle)
        
        edge_normal_src = torch.sum(direction * normal_src, dim=1, keepdim=True)
        edge_normal_dst = torch.sum(direction * normal_dst, dim=1, keepdim=True)
        edge_features.extend([edge_normal_src, edge_normal_dst])
    
    return torch.cat(edge_features, dim=1)

def add_multiscale_edges(data, scales=[2, 4, 8]):
    """Add multi-scale edges for hierarchical message passing."""
    edge_lists = [data.edge_index]
    
    pos = data.x[:, :3]
    num_nodes = pos.size(0)
    
    for scale in scales:
        sampled_indices = torch.arange(0, num_nodes, scale)
        
        if len(sampled_indices) > 1:
            sampled_pos = pos[sampled_indices]
            dist_matrix = torch.cdist(sampled_pos, sampled_pos)
            
            k = min(8, len(sampled_indices) - 1)
            _, knn_indices = torch.topk(dist_matrix, k + 1, dim=1, largest=False)
            knn_indices = knn_indices[:, 1:]
            
            src_nodes = sampled_indices.repeat_interleave(k)
            dst_nodes = sampled_indices[knn_indices.flatten()]
            
            scale_edges = torch.stack([src_nodes, dst_nodes])
            edge_lists.append(scale_edges)
    
    all_edges = torch.cat(edge_lists, dim=1)
    all_edges = torch.unique(all_edges, dim=1)
    
    return all_edges

# =============================================================================
# SURFACE PROPERTIES
# =============================================================================

def compute_surface_properties(data: Any, estimate_normals: bool = True, 
                             estimate_areas: bool = True) -> Any:
    """Compute surface properties (normals and areas) for graph data."""
    if not hasattr(data, 'x') or data.x.shape[1] < 3:
        warnings.warn("Data must have 3D coordinates to compute surface properties")
        return data
    
    pos = data.x[:, :3]
    
    if estimate_normals and not hasattr(data, 'normals'):
        try:
            normals = estimate_surface_normals(pos, data.edge_index)
            data.normals = normals
        except Exception as e:
            warnings.warn(f"Failed to estimate normals: {e}")
    
    if estimate_areas and not hasattr(data, 'area'):
        try:
            areas = estimate_surface_areas(pos, data.edge_index)
            data.area = areas
        except Exception as e:
            warnings.warn(f"Failed to estimate areas: {e}")
    
    return data

def estimate_surface_normals(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Estimate surface normals using local surface fitting."""
    num_nodes = pos.shape[0]
    normals = torch.zeros_like(pos)
    src, dst = edge_index
    
    for i in range(num_nodes):
        neighbors_mask = (src == i) | (dst == i)
        if not neighbors_mask.any():
            normals[i] = torch.tensor([0., 0., 1.])
            continue
        
        neighbor_edges = edge_index[:, neighbors_mask]
        neighbor_indices = torch.cat([
            neighbor_edges[1][neighbor_edges[0] == i],
            neighbor_edges[0][neighbor_edges[1] == i]
        ]).unique()
        
        if len(neighbor_indices) < 2:
            normals[i] = torch.tensor([0., 0., 1.])
            continue
        
        local_pos = pos[neighbor_indices] - pos[i:i+1]
        
        if local_pos.shape[0] >= 3:
            _, _, V = torch.svd(local_pos.T)
            normal = V[:, -1]
        else:
            if local_pos.shape[0] == 2:
                v1, v2 = local_pos[0], local_pos[1]
                normal = torch.cross(v1, v2)
                if normal.norm() < 1e-6:
                    normal = torch.tensor([0., 0., 1.])
                else:
                    normal = normal / normal.norm()
            else:
                normal = torch.tensor([0., 0., 1.])
        
        normals[i] = normal
    
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    return normals.to(torch.float32)

def estimate_surface_areas(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Estimate surface areas using Voronoi-like weighting."""
    num_nodes = pos.shape[0]
    areas = torch.zeros(num_nodes, 1, dtype=torch.float32)
    src, dst = edge_index
    
    edge_vectors = pos[dst] - pos[src]
    edge_lengths = torch.norm(edge_vectors, dim=1)
    
    for i in range(num_nodes):
        connected_edges = (src == i) | (dst == i)
        if connected_edges.any():
            local_edge_lengths = edge_lengths[connected_edges]
            estimated_area = (local_edge_lengths.mean() ** 2) * 0.5
            areas[i, 0] = estimated_area
        else:
            areas[i, 0] = 1.0
    
    return areas

# =============================================================================
# FILE I/O AND DATA MANAGEMENT
# =============================================================================

def load_graphs_from_directory(directory_path: AnyPath, pattern: str = 'graph_*.pt', 
                             max_graphs: Optional[int] = None, 
                             validate: bool = True) -> List[Any]:
    """Load graph files from directory with comprehensive error handling."""
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    graph_files = sorted(directory.glob(pattern))
    if max_graphs is not None:
        graph_files = graph_files[:max_graphs]
    
    if not graph_files:
        raise FileNotFoundError(f"No graph files found in {directory} with pattern {pattern}")
    
    print(f"📂 Found {len(graph_files)} graph files in {directory}")
    
    graphs = []
    failed_files = []
    
    pbar = create_progress_bar(len(graph_files), "Loading graphs")
    
    for file_path in graph_files:
        try:
            graph = torch.load(file_path, map_location='cpu', weights_only=False)
            
            if validate:
                validation_result = validate_graph_data(graph)
                if not validation_result['is_valid']:
                    warnings.warn(f"Invalid graph in {file_path.name}: "
                                f"missing {validation_result['missing_required']}")
            
            graphs.append(graph)
            
        except Exception as e:
            failed_files.append(file_path.name)
            warnings.warn(f"Failed to load {file_path.name}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    if failed_files:
        print(f"⚠️ Failed to load {len(failed_files)} files")
        if len(failed_files) <= 5:
            print(f"   Failed files: {failed_files}")
        else:
            print(f"   Failed files: {failed_files[:5]}... and {len(failed_files)-5} more")
    
    print(f"✅ Successfully loaded {len(graphs)} graphs")
    return graphs

def load_graphs_with_progress(save_dir="./saved_graphs"):
    """Load saved graph data with progress tracking."""
    save_dir = Path(save_dir)
    
    if not save_dir.exists():
        raise FileNotFoundError(f"Save directory not found: {save_dir}")
    
    print("Loading graph data...")
    
    total_steps = 2
    train_graphs = None
    val_graphs = None
    
    with tqdm(total=total_steps, desc="Loading datasets") as pbar:
        train_path = save_dir / "train_graphs.pt"
        if train_path.exists():
            train_graphs = torch.load(train_path, weights_only=False)
            pbar.set_description("Loaded train graphs")
            pbar.update(1)
        else:
            raise FileNotFoundError(f"Train graphs file not found: {train_path}")
        
        val_path = save_dir / "val_graphs.pt"
        if val_path.exists():
            val_graphs = torch.load(val_path, weights_only=False)
            pbar.set_description("Loaded validation graphs")
            pbar.update(1)
        else:
            raise FileNotFoundError(f"Validation graphs file not found: {val_path}")
    
    print(f"\n✅ Graphs loaded successfully!")
    print(f"📊 Train graphs: {len(train_graphs)} samples")
    print(f"📊 Validation graphs: {len(val_graphs)} samples")
    
    return train_graphs, val_graphs

def save_graphs_to_directory(graphs: List[Any], directory_path: AnyPath, 
                           prefix: str = 'graph', compress: bool = False) -> None:
    """Save graphs to individual files in directory."""
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    
    pbar = create_progress_bar(len(graphs), "Saving graphs")
    
    for i, graph in enumerate(graphs):
        filename = f"{prefix}_{i:06d}.pt"
        filepath = directory / filename
        
        try:
            if compress:
                torch.save(graph, filepath, pickle_protocol=4)
            else:
                torch.save(graph, filepath)
        except Exception as e:
            warnings.warn(f"Failed to save graph {i}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    print(f"💾 Saved {len(graphs)} graphs to {directory}")

def load_indices_packed(save_path: AnyPath, *, progress=True, position=3) -> List[torch.Tensor]:
    """Load packed node indices from cache file."""
    obj = torch.load(save_path, map_location='cpu', weights_only=False)
    ptr, nodes = obj["ptr"], obj["nodes"]
    out: List[torch.Tensor] = []

    bar = _mkbar(
        ptr.numel() - 1,
        "[cache] load patches",
        position=position,
        leave=False,
        progress=progress
    )

    for i in range(ptr.numel() - 1):
        lo, hi = ptr[i].item(), ptr[i + 1].item()
        out.append(nodes[lo:hi].clone())
        _update(bar, 1)
    _close(bar)
    return out

# =============================================================================
# DATASET STATISTICS
# =============================================================================

def compute_dataset_statistics(graphs: List[Any], verbose: bool = True) -> Dict[str, Any]:
    """Compute comprehensive statistics for a dataset of graphs."""
    if not graphs:
        return {}
    
    stats = {
        'num_graphs': len(graphs),
        'node_counts': [],
        'edge_counts': [],
        'feature_dims': {},
        'target_dims': {},
    }
    
    for graph in graphs:
        if hasattr(graph, 'x'):
            stats['node_counts'].append(graph.x.shape[0])
            if 'x' not in stats['feature_dims']:
                stats['feature_dims']['x'] = graph.x.shape[1]
        
        if hasattr(graph, 'edge_index'):
            stats['edge_counts'].append(graph.edge_index.shape[1])
        
        if hasattr(graph, 'y'):
            if 'y' not in stats['target_dims']:
                stats['target_dims']['y'] = graph.y.shape[1] if graph.y.dim() > 1 else 1
        
        if hasattr(graph, 'y_graph'):
            if 'y_graph' not in stats['target_dims']:
                stats['target_dims']['y_graph'] = (graph.y_graph.shape[1] 
                                                 if graph.y_graph.dim() > 1 else 1)
    
    # Compute summary statistics
    if stats['node_counts']:
        stats['nodes'] = {
            'mean': float(torch.tensor(stats['node_counts'], dtype=torch.float32).mean()),
            'std': float(torch.tensor(stats['node_counts'], dtype=torch.float32).std()),
            'min': min(stats['node_counts']),
            'max': max(stats['node_counts']),
            'total': sum(stats['node_counts'])
        }
    
    if stats['edge_counts']:
        stats['edges'] = {
            'mean': float(torch.tensor(stats['edge_counts'], dtype=torch.float32).mean()),
            'std': float(torch.tensor(stats['edge_counts'], dtype=torch.float32).std()),
            'min': min(stats['edge_counts']),
            'max': max(stats['edge_counts']),
            'total': sum(stats['edge_counts'])
        }
    
    if verbose:
        print(f"\n📊 Dataset Statistics:")
        print(f"  📈 Graphs: {stats['num_graphs']}")
        
        if 'nodes' in stats:
            print(f"  🔗 Nodes: {stats['nodes']['mean']:.1f}±{stats['nodes']['std']:.1f} "
                  f"(range: {stats['nodes']['min']}-{stats['nodes']['max']}, "
                  f"total: {stats['nodes']['total']:,})")
        
        if 'edges' in stats:
            print(f"  🔗 Edges: {stats['edges']['mean']:.1f}±{stats['edges']['std']:.1f} "
                  f"(range: {stats['edges']['min']}-{stats['edges']['max']}, "
                  f"total: {stats['edges']['total']:,})")
        
        if stats['feature_dims']:
            print(f"  📊 Feature dims: {stats['feature_dims']}")
        
        if stats['target_dims']:
            print(f"  🎯 Target dims: {stats['target_dims']}")
    
    return stats


def compute_relative_l2_error(pred, target, eps=1e-8):
    """
    Compute relative L2 error: ||pred - target||_2 / ||target||_2
    
    Args:
        pred: Predicted values [N, D]
        target: Ground truth values [N, D]
        eps: Small value to avoid division by zero
    
    Returns:
        relative_l2: Relative L2 error (scalar)
        per_channel_error: Relative L2 error per output channel [D]
    """
    # Ensure same device
    pred = pred.to(target.device)
    
    # Overall relative L2 error
    error_norm = torch.norm(pred - target, p=2)
    target_norm = torch.norm(target, p=2)
    relative_l2 = error_norm / (target_norm + eps)
    
    # Per-channel relative L2 error
    per_channel_error = []
    for i in range(target.shape[-1]):
        channel_error = torch.norm(pred[..., i] - target[..., i], p=2)
        channel_norm = torch.norm(target[..., i], p=2)
        per_channel_error.append((channel_error / (channel_norm + eps)).item())
    
    return relative_l2, torch.tensor(per_channel_error)

def compute_node_wise_relative_error(pred, target, eps=1e-8):
    """
    Compute node-wise relative L2 error for detailed analysis
    
    Args:
        pred: Predicted values [N, D]
        target: Ground truth values [N, D]
    
    Returns:
        node_errors: Relative error per node [N]
        mean_error: Mean relative error across nodes
        max_error: Maximum relative error
        percentiles: 25th, 50th, 75th, 95th percentiles
    """
    # Node-wise L2 norm of error
    node_errors = torch.norm(pred - target, p=2, dim=-1)
    node_targets = torch.norm(target, p=2, dim=-1)
    relative_errors = node_errors / (node_targets + eps)
    
    # Statistics
    mean_error = relative_errors.mean()
    max_error = relative_errors.max()
    
    # Percentiles
    percentiles = torch.quantile(relative_errors, 
                                 torch.tensor([0.25, 0.5, 0.75, 0.95], 
                                            device=relative_errors.device))
    
    return relative_errors, mean_error, max_error, percentiles

def compute_graph_level_error(pred, target, batch, eps=1e-8):
    """
    Compute graph-level relative L2 error for batch processing
    
    Args:
        pred: Predicted values [N, D]
        target: Ground truth values [N, D]
        batch: Batch assignment tensor [N]
    
    Returns:
        graph_errors: Relative L2 error per graph in batch
    """
    graph_errors = []
    
    for graph_idx in torch.unique(batch):
        mask = batch == graph_idx
        graph_pred = pred[mask]
        graph_target = target[mask]
        
        error_norm = torch.norm(graph_pred - graph_target, p=2)
        target_norm = torch.norm(graph_target, p=2)
        rel_error = error_norm / (target_norm + eps)
        graph_errors.append(rel_error.item())
    
    return torch.tensor(graph_errors)

def compute_epoch_relative_error(model, data_loader, device, use_ensemble=None):
    from model import EnsemblePredictor
    
    """
    Compute average relative L2 error over entire epoch
    
    Args:
        model: Model to evaluate (single or ensemble)
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        use_ensemble: Whether to use ensemble mode (auto-detect if None)
    
    Returns:
        avg_rel_l2: Average relative L2 error
        std_rel_l2: Standard deviation of relative L2 errors
        per_channel_avg: Average per-channel errors
        all_errors: List of all batch errors for statistics
    """
    model.eval()
    all_rel_l2 = []
    all_per_channel = []
    
    # Auto-detect ensemble mode if not specified
    if use_ensemble is None:
        use_ensemble = isinstance(model, EnsemblePredictor)
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Handle both single model and ensemble
            if use_ensemble:
                pred_mean, _ = model(batch)
            else:
                pred_mean = model(batch)
            
            rel_l2, per_channel = compute_relative_l2_error(pred_mean, batch.y)
            all_rel_l2.append(rel_l2.item())
            all_per_channel.append(per_channel)
    
    avg_rel_l2 = np.mean(all_rel_l2)
    std_rel_l2 = np.std(all_rel_l2)
    per_channel_avg = torch.stack(all_per_channel).mean(dim=0)
    
    return avg_rel_l2, std_rel_l2, per_channel_avg, all_rel_l2



# ============================================
# Memory Management Utilities
# ============================================

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"💾 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

def clear_gpu_cache():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU cache cleared")

# ============================================
# WandB Integration Utilities
# ============================================

class WandBLogger:
    """WandB integration for experiment tracking and logging"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.wandb = None
        self.run = None
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                warnings.warn("WandB not available. Install with: pip install wandb")
                self.enabled = False
    
    def init(self, project_name, experiment_name=None, config=None, tags=None, notes=None):
        """Initialize WandB run"""
        if not self.enabled:
            return False
            
        try:
            self.run = self.wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                tags=tags,
                notes=notes,
                reinit=True
            )
            print(f"🚀 WandB initialized: {self.run.url}")
            return True
        except Exception as e:
            warnings.warn(f"Failed to initialize WandB: {e}")
            self.enabled = False
            return False
    
    def log(self, metrics, step=None, commit=True):
        """Log metrics to WandB"""
        if not self.enabled or not self.run:
            return
        
        try:
            self.wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            warnings.warn(f"Failed to log to WandB: {e}")
    
    def log_model_architecture(self, model):
        """Log model architecture and parameters"""
        if not self.enabled or not self.run:
            return
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/model_size_mb": total_params * 4 / 1e6  # Assuming float32
            })
            
            # Log model graph if possible
            try:
                self.wandb.watch(model, log="all", log_freq=100)
            except:
                pass  # Some models might not be compatible with watch
                
        except Exception as e:
            warnings.warn(f"Failed to log model architecture: {e}")
    
    def log_training_metrics(self, epoch, train_loss, train_rel_l2, val_rel_l2=None, 
                           learning_rate=None, gpu_memory=None, train_per_channel=None, 
                           val_per_channel=None, uncertainty_stats=None):
        """Log comprehensive training metrics"""
        if not self.enabled or not self.run:
            return
        
        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/rel_l2_error": train_rel_l2,
        }
        
        if val_rel_l2 is not None:
            metrics["val/rel_l2_error"] = val_rel_l2
        
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate
        
        if gpu_memory is not None:
            metrics["system/gpu_memory_gb"] = gpu_memory
        
        # Log per-channel errors
        if train_per_channel is not None:
            for i, error in enumerate(train_per_channel):
                metrics[f"train/rel_l2_channel_{i}"] = error
        
        if val_per_channel is not None:
            for i, error in enumerate(val_per_channel):
                metrics[f"val/rel_l2_channel_{i}"] = error
        
        # Log uncertainty statistics if available (ensemble mode)
        if uncertainty_stats is not None:
            for key, value in uncertainty_stats.items():
                metrics[f"uncertainty/{key}"] = value
        
        self.log(metrics, step=epoch)
    
    def log_loss_components(self, epoch, mse_loss, physics_loss, smoothness_loss, 
                          loss_weights=None):
        """Log individual loss components"""
        if not self.enabled or not self.run:
            return
        
        metrics = {
            "losses/mse": mse_loss,
            "losses/physics": physics_loss,
            "losses/smoothness": smoothness_loss,
        }
        
        if loss_weights:
            for key, weight in loss_weights.items():
                metrics[f"loss_weights/{key}"] = weight
        
        self.log(metrics, step=epoch, commit=False)
    
    def log_inference_results(self, inference_time, uncertainty_stats=None, 
                            error_stats=None):
        """Log inference performance and results"""
        if not self.enabled or not self.run:
            return
        
        metrics = {
            "inference/time_ms": inference_time,
        }
        
        if uncertainty_stats:
            for key, value in uncertainty_stats.items():
                metrics[f"inference/uncertainty_{key}"] = value
        
        if error_stats:
            for key, value in error_stats.items():
                metrics[f"inference/error_{key}"] = value
        
        self.log(metrics)
    
    def log_memory_usage(self, system_memory=None, gpu_memory=None, memory_delta=None):
        """Log memory usage statistics"""
        if not self.enabled or not self.run:
            return
        
        metrics = {}
        
        if system_memory:
            metrics.update({
                "memory/system_used_gb": system_memory.get('used', 0) / 1e9,
                "memory/system_available_gb": system_memory.get('available', 0) / 1e9,
                "memory/system_percent": system_memory.get('percent', 0)
            })
        
        if gpu_memory:
            metrics.update({
                "memory/gpu_allocated_gb": gpu_memory.get('allocated', 0) / 1e9,
                "memory/gpu_cached_gb": gpu_memory.get('cached', 0) / 1e9,
                "memory/gpu_free_gb": gpu_memory.get('free', 0) / 1e9,
                "memory/gpu_percent": gpu_memory.get('percent', 0)
            })
        
        if memory_delta:
            for key, value in memory_delta.items():
                metrics[f"memory/delta_{key}"] = value
        
        if metrics:
            self.log(metrics, commit=False)
    
    def log_hyperparameters(self, config):
        """Log hyperparameters and configuration"""
        if not self.enabled or not self.run:
            return
        
        try:
            # Flatten nested config for better visualization
            flat_config = {}
            for key, value in config.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_config[f"{key}/{subkey}"] = subvalue
                else:
                    flat_config[key] = value
            
            self.wandb.config.update(flat_config)
        except Exception as e:
            warnings.warn(f"Failed to log hyperparameters: {e}")
    
    def finish(self):
        """Finish WandB run"""
        if self.enabled and self.run:
            self.wandb.finish()
            print("✅ WandB run finished")


def setup_wandb_logging(project_name, experiment_name=None, config=None, 
                       tags=None, notes=None, enabled=True):
    """
    Setup WandB logging with configuration
    
    Args:
        project_name: WandB project name
        experiment_name: Name for this specific experiment
        config: Configuration dictionary to log
        tags: List of tags for the experiment
        notes: Description or notes for the experiment
        enabled: Whether to enable WandB logging
    
    Returns:
        WandBLogger instance
    """
    logger = WandBLogger(enabled=enabled)
    
    if enabled:
        success = logger.init(
            project_name=project_name,
            experiment_name=experiment_name,
            config=config,
            tags=tags,
            notes=notes
        )
        
        if not success:
            print("⚠️ WandB logging disabled due to initialization failure")
    else:
        print("ℹ️ WandB logging disabled")
    
    return logger

        