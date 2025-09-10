"""
Simplified loss function for debugging
Start with MSE only, then gradually add physics terms
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

def compute_simple_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                       loss_type: str = 'mse_only') -> Dict[str, torch.Tensor]:
    """
    Compute simplified loss for debugging
    
    Args:
        predictions: [N, 4] tensor [pressure_coeff, tau_x, tau_y, tau_z]
        targets: [N, 4] tensor with same structure
        loss_type: 'mse_only', 'mse_plus_smooth', or 'mse_plus_physics'
    
    Returns:
        Dictionary with loss components
    """
    
    device = predictions.device
    loss_components = {}
    
    # 1. Basic MSE Loss (always included)
    mse_loss = F.mse_loss(predictions, targets)
    loss_components['mse'] = mse_loss
    loss_components['total_loss'] = mse_loss
    
    if loss_type == 'mse_only':
        # Only MSE loss
        return loss_components
    
    elif loss_type == 'mse_plus_smooth':
        # MSE + simple smoothness
        try:
            if len(predictions) > 1:
                # Simple smoothness: penalize large changes between adjacent predictions
                smooth_loss = 0.0
                for i in range(4):  # For each output component
                    diff = torch.diff(predictions[:, i])
                    smooth_loss += torch.mean(diff**2)
                smooth_loss = smooth_loss / 4
                
                loss_components['smoothness'] = smooth_loss
                loss_components['total_loss'] = mse_loss + 0.1 * smooth_loss
            else:
                loss_components['smoothness'] = torch.tensor(0.0, device=device)
        except Exception as e:
            print(f"Smoothness loss failed: {e}")
            loss_components['smoothness'] = torch.tensor(0.0, device=device)
    
    elif loss_type == 'mse_plus_physics':
        # MSE + simple physics constraints
        try:
            # Simple pressure constraint: reasonable range
            p_coeff = predictions[:, 0]
            pressure_penalty = F.relu(torch.abs(p_coeff) - 5.0).mean()  # Penalty for |Cp| > 5
            
            # Simple shear stress constraint: reasonable magnitude
            tau_magnitude = torch.sqrt(predictions[:, 1]**2 + predictions[:, 2]**2 + predictions[:, 3]**2 + 1e-8)
            shear_penalty = F.relu(tau_magnitude - 10.0).mean()  # Penalty for |tau| > 10
            
            # Simple correlation: pressure and shear should be somewhat related
            if len(p_coeff) > 1:
                try:
                    corr_matrix = torch.corrcoef(torch.stack([p_coeff, tau_magnitude]))
                    correlation_loss = 1.0 - torch.abs(corr_matrix[0, 1])
                except:
                    correlation_loss = torch.tensor(0.0, device=device)
            else:
                correlation_loss = torch.tensor(0.0, device=device)
            
            physics_loss = pressure_penalty + shear_penalty + 0.1 * correlation_loss
            
            loss_components['pressure_penalty'] = pressure_penalty
            loss_components['shear_penalty'] = shear_penalty
            loss_components['correlation_loss'] = correlation_loss
            loss_components['physics_loss'] = physics_loss
            loss_components['total_loss'] = mse_loss + 0.1 * physics_loss
            
        except Exception as e:
            print(f"Physics loss failed: {e}")
            loss_components['physics_loss'] = torch.tensor(0.0, device=device)
    
    return loss_components

def compute_debug_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                      data: Any, debug_level: int = 1) -> Dict[str, torch.Tensor]:
    """
    Debug version of comprehensive loss
    
    debug_level:
        1: MSE only
        2: MSE + smoothness
        3: MSE + physics
    """
    
    if debug_level == 1:
        return compute_simple_loss(predictions, targets, 'mse_only')
    elif debug_level == 2:
        return compute_simple_loss(predictions, targets, 'mse_plus_smooth')
    elif debug_level == 3:
        return compute_simple_loss(predictions, targets, 'mse_plus_physics')
    else:
        # Fallback to MSE only
        return compute_simple_loss(predictions, targets, 'mse_only')

# Test the simple loss functions
if __name__ == "__main__":
    print("🧪 Testing simple loss functions...")
    
    # Create test data
    predictions = torch.randn(10, 4)
    targets = torch.randn(10, 4)
    
    # Test different loss types
    for loss_type in ['mse_only', 'mse_plus_smooth', 'mse_plus_physics']:
        result = compute_simple_loss(predictions, targets, loss_type)
        print(f"\\n{loss_type}:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
    
    print("\\n✅ Simple loss functions working!")