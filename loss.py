import torch
import torch.nn.functional as F

# ============================================
# Physics Loss Functions (CUDA-aware)
# ============================================

def compute_physics_loss(pred, batch):
    """Compute physics-informed loss (simplified example)"""
    loss = torch.zeros(1, device=pred.device)
    
    if hasattr(batch, 'pos'):
        pos = batch.pos.to(pred.device)
        edge_index = batch.edge_index.to(pred.device)
        
        src, dst = edge_index
        pred_diff = pred[src] - pred[dst]
        pos_diff = pos[src] - pos[dst]
        dist = torch.norm(pos_diff, dim=1, keepdim=True) + 1e-8
        
        # Weighted smoothness by inverse distance
        loss = (pred_diff.pow(2) / dist).mean()
    
    return loss

def compute_smoothness_loss(pred, edge_index):
    """Compute smoothness loss over edges"""
    edge_index = edge_index.to(pred.device)
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    return diff.pow(2).mean()

# ============================================
# PINN Loss Functions with Pressure-WSS Coupling
# ============================================

def compute_pressure_gradient(pressure, pos, edge_index, eps=1e-8):
    """
    Compute pressure gradient along surface edges
    
    Args:
        pressure: Pressure field [N, 1] or [N]
        pos: Node positions [N, 3] or [N, 2]
        edge_index: Edge connectivity [2, E]
        eps: Small value to avoid division by zero
        
    Returns:
        pressure_grad: Pressure gradient magnitude per edge [E]
        pressure_grad_vec: Pressure gradient vector per edge [E, spatial_dim]
    """
    if pressure.dim() == 2 and pressure.shape[1] == 1:
        pressure = pressure.squeeze(-1)
    
    src, dst = edge_index
    
    # Pressure difference
    dp = pressure[dst] - pressure[src]
    
    # Spatial difference
    dx = pos[dst] - pos[src]  # [E, spatial_dim]
    dist = torch.norm(dx, dim=1, keepdim=True) + eps  # [E, 1]
    
    # Pressure gradient magnitude
    pressure_grad_mag = torch.abs(dp) / dist.squeeze(-1)  # [E]
    
    # Pressure gradient vector (normalized direction)
    direction = dx / dist  # [E, spatial_dim]
    pressure_grad_vec = dp.unsqueeze(-1) * direction / dist  # [E, spatial_dim]
    
    return pressure_grad_mag, pressure_grad_vec

def compute_wall_shear_stress(velocity_pred, pos, edge_index, wall_normal=None, eps=1e-8):
    """
    Compute wall shear stress from velocity predictions
    
    Args:
        velocity_pred: Velocity predictions [N, 3] or [N, 2]
        pos: Node positions [N, 3] or [N, 2] 
        edge_index: Edge connectivity [2, E]
        wall_normal: Wall normal vectors [N, 3] or [N, 2], if available
        eps: Small value to avoid division by zero
        
    Returns:
        wss_magnitude: WSS magnitude per node [N]
        wss_vector: WSS vector per node [N, spatial_dim]
    """
    num_nodes = velocity_pred.shape[0]
    spatial_dim = velocity_pred.shape[1]
    
    src, dst = edge_index
    
    # Velocity difference across edges
    du = velocity_pred[dst] - velocity_pred[src]  # [E, spatial_dim]
    
    # Spatial difference
    dx = pos[dst] - pos[src]  # [E, spatial_dim]
    dist = torch.norm(dx, dim=1, keepdim=True) + eps  # [E, 1]
    
    # Velocity gradient approximation: du/dx
    velocity_grad = du / dist  # [E, spatial_dim]
    
    # Aggregate velocity gradients to nodes (average over connected edges)
    node_velocity_grad = torch.zeros(num_nodes, spatial_dim, device=velocity_pred.device)
    node_count = torch.zeros(num_nodes, 1, device=velocity_pred.device)
    
    # Add contributions from both source and destination nodes
    node_velocity_grad.index_add_(0, dst, velocity_grad)
    node_velocity_grad.index_add_(0, src, -velocity_grad)  # Negative for reverse direction
    node_count.index_add_(0, dst, torch.ones_like(dist))
    node_count.index_add_(0, src, torch.ones_like(dist))
    
    # Average the gradients
    node_velocity_grad = node_velocity_grad / (node_count + eps)
    
    # Compute WSS magnitude (simplified as velocity gradient magnitude)
    wss_magnitude = torch.norm(node_velocity_grad, dim=1)  # [N]
    
    # If wall normals are provided, compute wall-tangential component
    if wall_normal is not None:
        # WSS = velocity_gradient - (velocity_gradient · normal) * normal
        normal_component = torch.sum(node_velocity_grad * wall_normal, dim=1, keepdim=True)
        wss_vector = node_velocity_grad - normal_component * wall_normal
        wss_magnitude = torch.norm(wss_vector, dim=1)
        return wss_magnitude, wss_vector
    
    return wss_magnitude, node_velocity_grad

def pressure_wss_coupling_loss(pred, pos, edge_index, 
                              pressure_weight=1.0, wss_weight=1.0,
                              separation_threshold=0.1, attachment_threshold=2.0,
                              eps=1e-8):
    """
    Physics-informed loss enforcing pressure gradient and WSS relationship
    
    Physical relationships:
    - Adverse pressure gradient (dp/dx > 0) → low WSS (flow separation)
    - Favorable pressure gradient (dp/dx < 0) → high WSS (flow attachment)
    
    Args:
        pred: Model predictions [N, output_dim]
               Assumes: [pressure, velocity_x, velocity_y, (velocity_z)]
        pos: Node positions [N, spatial_dim]
        edge_index: Edge connectivity [2, E]
        pressure_weight: Weight for pressure gradient penalty
        wss_weight: Weight for WSS penalty
        separation_threshold: WSS threshold below which flow is considered separated
        attachment_threshold: WSS threshold above which flow is considered attached
        eps: Small value for numerical stability
        
    Returns:
        loss: Physics-informed coupling loss
        loss_components: Dictionary with individual loss components
    """
    if pred.shape[1] < 3:
        raise ValueError("Predictions must include at least pressure and 2D velocity")
    
    # Extract pressure and velocity
    pressure = pred[:, 0]  # [N]
    velocity = pred[:, 1:4] if pred.shape[1] >= 4 else pred[:, 1:3]  # [N, spatial_dim]
    
    # Compute pressure gradients
    pressure_grad_mag, pressure_grad_vec = compute_pressure_gradient(
        pressure, pos, edge_index, eps=eps
    )
    
    # Compute wall shear stress
    wss_magnitude, wss_vector = compute_wall_shear_stress(
        velocity, pos, edge_index, eps=eps
    )
    
    # Map edge-based pressure gradients to nodes (average)
    num_nodes = pred.shape[0]
    src, dst = edge_index
    
    node_pressure_grad = torch.zeros(num_nodes, device=pred.device)
    node_count = torch.zeros(num_nodes, device=pred.device)
    
    # Add pressure gradient contributions to connected nodes
    node_pressure_grad.index_add_(0, src, pressure_grad_mag)
    node_pressure_grad.index_add_(0, dst, pressure_grad_mag)
    node_count.index_add_(0, src, torch.ones_like(pressure_grad_mag))
    node_count.index_add_(0, dst, torch.ones_like(pressure_grad_mag))
    
    node_pressure_grad = node_pressure_grad / (node_count + eps)
    
    # Determine pressure gradient sign (simplified: use pressure gradient magnitude)
    # Positive gradient = adverse, Negative gradient = favorable
    adverse_mask = node_pressure_grad > eps  # Adverse pressure gradient regions
    favorable_mask = node_pressure_grad > eps  # For this implementation, we focus on magnitude
    
    # Physics-based penalties
    loss_components = {}
    
    # Penalty 1: Adverse pressure gradient should have low WSS
    adverse_penalty = torch.mean(
        adverse_mask.float() * torch.relu(wss_magnitude - separation_threshold) ** 2
    )
    
    # Penalty 2: High pressure gradient should correspond to appropriate WSS
    # For separation: high pressure gradient + high WSS is penalized
    separation_penalty = torch.mean(
        (node_pressure_grad ** 2) * torch.relu(wss_magnitude - separation_threshold) ** 2
    )
    
    # Penalty 3: Smoothness constraint on WSS field
    wss_smoothness = compute_smoothness_loss(wss_magnitude.unsqueeze(-1), edge_index)
    
    # Penalty 4: Pressure gradient smoothness
    pressure_grad_smoothness = torch.mean(
        (pressure_grad_mag[:-1] - pressure_grad_mag[1:]) ** 2
    )
    
    # Total loss
    total_loss = (
        pressure_weight * separation_penalty +
        wss_weight * adverse_penalty +
        0.1 * wss_smoothness +
        0.1 * pressure_grad_smoothness
    )
    
    loss_components.update({
        'separation_penalty': separation_penalty.item(),
        'adverse_penalty': adverse_penalty.item(),
        'wss_smoothness': wss_smoothness.item(),
        'pressure_grad_smoothness': pressure_grad_smoothness.item(),
        'total_physics_loss': total_loss.item()
    })
    
    return total_loss, loss_components

def advanced_pinn_loss(pred, batch, 
                      physics_weight=1.0, smoothness_weight=0.1,
                      pressure_weight=1.0, wss_weight=1.0,
                      include_original_physics=True):
    """
    Advanced PINN loss combining multiple physics constraints
    
    Args:
        pred: Model predictions [N, output_dim]
        batch: Batch data containing pos, edge_index, etc.
        physics_weight: Weight for physics-informed coupling loss
        smoothness_weight: Weight for smoothness regularization
        pressure_weight: Weight for pressure gradient penalties
        wss_weight: Weight for WSS penalties
        include_original_physics: Whether to include original physics loss
        
    Returns:
        total_loss: Combined physics-informed loss
        loss_dict: Dictionary with all loss components
    """
    device = pred.device
    loss_dict = {}
    
    # Original physics loss (if requested)
    if include_original_physics and hasattr(batch, 'pos'):
        original_physics_loss = compute_physics_loss(pred, batch)
        loss_dict['original_physics'] = original_physics_loss.item()
    else:
        original_physics_loss = torch.tensor(0.0, device=device)
        loss_dict['original_physics'] = 0.0
    
    # Smoothness regularization
    if hasattr(batch, 'edge_index'):
        smoothness_loss = compute_smoothness_loss(pred, batch.edge_index)
        loss_dict['smoothness'] = smoothness_loss.item()
    else:
        smoothness_loss = torch.tensor(0.0, device=device)
        loss_dict['smoothness'] = 0.0
    
    # Pressure-WSS coupling loss (main contribution)
    if hasattr(batch, 'pos') and hasattr(batch, 'edge_index') and pred.shape[1] >= 3:
        coupling_loss, coupling_components = pressure_wss_coupling_loss(
            pred, batch.pos, batch.edge_index,
            pressure_weight=pressure_weight,
            wss_weight=wss_weight
        )
        loss_dict.update(coupling_components)
    else:
        coupling_loss = torch.tensor(0.0, device=device)
        loss_dict.update({
            'separation_penalty': 0.0,
            'adverse_penalty': 0.0,
            'wss_smoothness': 0.0,
            'pressure_grad_smoothness': 0.0,
            'total_physics_loss': 0.0
        })
    
    # Combine all losses
    total_loss = (
        original_physics_loss +
        physics_weight * coupling_loss +
        smoothness_weight * smoothness_loss
    )
    
    loss_dict['total_loss'] = total_loss.item()
    
    return total_loss, loss_dict

# ============================================
# Individual PINN Loss Functions for Notebook Integration
# ============================================

def compute_pinn_loss(pred, batch, physics_weight=1.0, pressure_weight=0.5, wss_weight=0.5):
    """
    Main PINN loss function combining pressure-WSS physics
    
    Args:
        pred: Model predictions [N, output_dim]
        batch: Batch data with pos, edge_index
        physics_weight: Overall physics weight
        pressure_weight: Pressure gradient weight
        wss_weight: WSS penalty weight
        
    Returns:
        pinn_loss: Combined physics-informed loss
    """
    if not (hasattr(batch, 'pos') and hasattr(batch, 'edge_index') and pred.shape[1] >= 3):
        return torch.tensor(0.0, device=pred.device)
    
    # Use the advanced PINN loss with specific weights
    total_loss, _ = advanced_pinn_loss(
        pred, batch,
        physics_weight=physics_weight,
        pressure_weight=pressure_weight,
        wss_weight=wss_weight,
        include_original_physics=False  # Focus on pressure-WSS coupling
    )
    
    return total_loss

def compute_pressure_gradient_loss(pred, batch, threshold=0.1, smoothness_weight=0.1):
    """
    Pressure gradient consistency loss
    
    Args:
        pred: Model predictions [N, output_dim] (pressure in first column)
        batch: Batch data with pos, edge_index
        threshold: Threshold for pressure gradient penalties
        smoothness_weight: Weight for gradient smoothness
        
    Returns:
        pressure_loss: Pressure gradient loss
    """
    if not (hasattr(batch, 'pos') and hasattr(batch, 'edge_index') and pred.shape[1] >= 1):
        return torch.tensor(0.0, device=pred.device)
    
    try:
        pressure = pred[:, 0]  # Extract pressure
        
        # Compute pressure gradients
        pressure_grad_mag, _ = compute_pressure_gradient(
            pressure, batch.pos, batch.edge_index
        )
        
        # Penalty for excessive pressure gradients
        gradient_penalty = torch.mean(torch.relu(pressure_grad_mag - threshold) ** 2)
        
        # Smoothness penalty for pressure gradient field
        if len(pressure_grad_mag) > 1:
            gradient_smoothness = torch.mean(
                (pressure_grad_mag[:-1] - pressure_grad_mag[1:]) ** 2
            )
        else:
            gradient_smoothness = torch.tensor(0.0, device=pred.device)
        
        total_loss = gradient_penalty + smoothness_weight * gradient_smoothness
        
        return total_loss
        
    except Exception:
        # Fallback to simple pressure smoothness
        return compute_smoothness_loss(pred[:, 0:1], batch.edge_index)

def compute_wall_shear_stress_loss(pred, batch, separation_threshold=0.1, attachment_threshold=2.0):
    """
    Wall shear stress physics loss
    
    Args:
        pred: Model predictions [N, output_dim] (velocity in columns 1+)
        batch: Batch data with pos, edge_index
        separation_threshold: WSS threshold for flow separation
        attachment_threshold: WSS threshold for flow attachment
        
    Returns:
        wss_loss: Wall shear stress physics loss
    """
    if not (hasattr(batch, 'pos') and hasattr(batch, 'edge_index') and pred.shape[1] >= 3):
        return torch.tensor(0.0, device=pred.device)
    
    try:
        # Extract velocity components
        velocity = pred[:, 1:4] if pred.shape[1] >= 4 else pred[:, 1:3]
        
        # Compute wall shear stress
        wss_magnitude, _ = compute_wall_shear_stress(velocity, batch.pos, batch.edge_index)
        
        # Physics-based penalties
        # Penalty for unphysical WSS values (too high or too low)
        low_wss_penalty = torch.mean(torch.relu(separation_threshold - wss_magnitude) ** 2)
        high_wss_penalty = torch.mean(torch.relu(wss_magnitude - attachment_threshold) ** 2)
        
        # WSS field smoothness
        wss_smoothness = compute_smoothness_loss(wss_magnitude.unsqueeze(-1), batch.edge_index)
        
        total_loss = low_wss_penalty + 0.1 * high_wss_penalty + 0.1 * wss_smoothness
        
        return total_loss
        
    except Exception:
        # Fallback to velocity field smoothness
        velocity = pred[:, 1:4] if pred.shape[1] >= 4 else pred[:, 1:3]
        return compute_smoothness_loss(velocity, batch.edge_index)