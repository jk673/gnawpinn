"""
Corrected Comprehensive Loss Functions for Pressure and Wall Shear Stress Data
Based on actual data structure: [pressure_coefficient, tau_x, tau_y, tau_z]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class CorrectedPINNLoss(nn.Module):
    """Physics-informed loss for pressure coefficients and wall shear stress components"""
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Corrected loss weights for available data
        self.default_weights = {
            'mse': 1.0,                          # Primary data fitting loss
            'pressure_smoothness': 0.20,         # Pressure field smoothness
            'shear_stress_smoothness': 0.15,     # WSS field smoothness
            'shear_stress_magnitude': 0.12,      # WSS magnitude consistency
            'pressure_gradient': 0.18,           # Pressure gradient constraints
            'wall_shear_balance': 0.15,          # WSS component balance
            'boundary_pressure': 0.20,           # Pressure boundary conditions
            'boundary_shear': 0.15,              # Shear boundary conditions
            'physical_consistency': 0.10,        # Physical range constraints
            'multi_component_correlation': 0.08, # Correlation between components
            'spatial_coherence': 0.12,           # Spatial field coherence
        }
        
        self.loss_weights = loss_weights or self.default_weights.copy()
    
    def compute_spatial_gradients(self, predictions: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute spatial gradients for pressure and shear stress fields"""
        gradients = {}
        
        if predictions.requires_grad and pos.requires_grad:
            try:
                # predictions: [N, 4] -> [pressure_coeff, tau_x, tau_y, tau_z]
                
                # Pressure coefficient gradients
                p_coeff = predictions[:, 0:1]
                if pos.shape[1] >= 2:
                    dp_dx = torch.autograd.grad(
                        p_coeff.sum(), pos, create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    if dp_dx is not None:
                        gradients.update({'dp_dx': dp_dx[:, 0:1], 'dp_dy': dp_dx[:, 1:2]})
                
                # Wall shear stress gradients
                tau_x = predictions[:, 1:2]
                tau_y = predictions[:, 2:3] 
                
                if pos.shape[1] >= 2:
                    dtaux_dx = torch.autograd.grad(
                        tau_x.sum(), pos, create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    dtauy_dx = torch.autograd.grad(
                        tau_y.sum(), pos, create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    
                    if dtaux_dx is not None and dtauy_dx is not None:
                        gradients.update({
                            'dtaux_dx': dtaux_dx[:, 0:1], 'dtaux_dy': dtaux_dx[:, 1:2],
                            'dtauy_dx': dtauy_dx[:, 0:1], 'dtauy_dy': dtauy_dx[:, 1:2]
                        })
            except Exception as e:
                print(f"Gradient computation failed: {e}")
                pass
        
        return gradients
    
    def pressure_smoothness_loss(self, predictions: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Pressure field smoothness constraints"""
        try:
            gradients = self.compute_spatial_gradients(predictions, pos)
            
            if 'dp_dx' in gradients and 'dp_dy' in gradients:
                # Pressure gradient magnitude
                pressure_grad_mag = torch.sqrt(gradients['dp_dx']**2 + gradients['dp_dy']**2 + 1e-8)
                
                # Smoothness: penalize large variations in pressure gradient
                if len(pressure_grad_mag) > 1:
                    grad_variation = torch.diff(pressure_grad_mag, dim=0)
                    smoothness_loss = torch.mean(grad_variation**2)
                else:
                    smoothness_loss = torch.tensor(0.0, device=predictions.device)
                
                return {
                    'pressure_smoothness_loss': smoothness_loss,
                    'mean_pressure_gradient': torch.mean(pressure_grad_mag),
                    'max_pressure_gradient': torch.max(pressure_grad_mag)
                }
            else:
                # Fallback: simple pressure variation
                p_coeff = predictions[:, 0]
                if len(p_coeff) > 1:
                    p_variation = torch.diff(p_coeff)
                    smoothness_loss = torch.var(p_variation)
                else:
                    smoothness_loss = torch.tensor(0.0, device=predictions.device)
                
                return {
                    'pressure_smoothness_loss': smoothness_loss,
                    'mean_pressure_gradient': torch.tensor(0.0, device=predictions.device),
                    'max_pressure_gradient': torch.tensor(0.0, device=predictions.device)
                }
                
        except Exception as e:
            print(f"Warning: Pressure smoothness loss failed: {e}")
            return {
                'pressure_smoothness_loss': torch.tensor(0.0, device=predictions.device),
                'mean_pressure_gradient': torch.tensor(0.0, device=predictions.device),
                'max_pressure_gradient': torch.tensor(0.0, device=predictions.device)
            }
    
    def shear_stress_losses(self, predictions: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Wall shear stress consistency and smoothness losses"""
        try:
            tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
            
            # 1. Shear stress magnitude consistency
            tau_magnitude = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-8)
            
            # 2. Smoothness of individual components
            smoothness_losses = []
            if len(tau_x) > 1:
                smoothness_losses.extend([
                    torch.mean(torch.diff(tau_x)**2),
                    torch.mean(torch.diff(tau_y)**2),
                    torch.mean(torch.diff(tau_z)**2)
                ])
            
            smoothness_loss = torch.stack(smoothness_losses).mean() if smoothness_losses else torch.tensor(0.0, device=predictions.device)
            
            # 3. Magnitude smoothness
            if len(tau_magnitude) > 1:
                magnitude_smoothness = torch.mean(torch.diff(tau_magnitude)**2)
            else:
                magnitude_smoothness = torch.tensor(0.0, device=predictions.device)
            
            # 4. Component balance (no single component dominates unrealistically)
            component_balance = torch.var(torch.stack([torch.mean(torch.abs(tau_x)), 
                                                     torch.mean(torch.abs(tau_y)), 
                                                     torch.mean(torch.abs(tau_z))]))
            
            return {
                'shear_stress_smoothness_loss': smoothness_loss,
                'shear_stress_magnitude_loss': magnitude_smoothness,
                'shear_component_balance_loss': component_balance,
                'mean_shear_magnitude': torch.mean(tau_magnitude),
                'max_shear_magnitude': torch.max(tau_magnitude),
                'shear_x_rms': torch.sqrt(torch.mean(tau_x**2)),
                'shear_y_rms': torch.sqrt(torch.mean(tau_y**2)),
                'shear_z_rms': torch.sqrt(torch.mean(tau_z**2))
            }
            
        except Exception as e:
            print(f"Warning: Shear stress loss computation failed: {e}")
            return {
                'shear_stress_smoothness_loss': torch.tensor(0.0, device=predictions.device),
                'shear_stress_magnitude_loss': torch.tensor(0.0, device=predictions.device),
                'shear_component_balance_loss': torch.tensor(0.0, device=predictions.device),
                'mean_shear_magnitude': torch.tensor(0.0, device=predictions.device),
                'max_shear_magnitude': torch.tensor(0.0, device=predictions.device),
                'shear_x_rms': torch.tensor(0.0, device=predictions.device),
                'shear_y_rms': torch.tensor(0.0, device=predictions.device),
                'shear_z_rms': torch.tensor(0.0, device=predictions.device)
            }
    
    def wall_shear_balance_loss(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enforce physical balance between shear stress components"""
        try:
            tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
            
            # Physical constraint: shear stress should have coherent directional patterns
            # No single component should be orders of magnitude larger than others consistently
            
            tau_magnitude = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-8)
            
            # Normalized components
            tau_x_norm = tau_x / (tau_magnitude + 1e-8)
            tau_y_norm = tau_y / (tau_magnitude + 1e-8)
            tau_z_norm = tau_z / (tau_magnitude + 1e-8)
            
            # Balance loss: prevent extreme dominance of single component
            component_ratios = torch.stack([
                torch.abs(tau_x_norm),
                torch.abs(tau_y_norm), 
                torch.abs(tau_z_norm)
            ], dim=1)
            
            # Penalize when one component is consistently >> others
            max_component = torch.max(component_ratios, dim=1)[0]
            balance_loss = torch.mean((max_component - 0.577)**2)  # 0.577 = 1/sqrt(3) for balanced case
            
            return {
                'wall_shear_balance_loss': balance_loss,
                'mean_max_component_ratio': torch.mean(max_component),
                'component_balance_std': torch.std(max_component)
            }
            
        except Exception as e:
            print(f"Warning: Wall shear balance loss failed: {e}")
            return {
                'wall_shear_balance_loss': torch.tensor(0.0, device=predictions.device),
                'mean_max_component_ratio': torch.tensor(0.0, device=predictions.device),
                'component_balance_std': torch.tensor(0.0, device=predictions.device)
            }
    
    def physical_consistency_loss(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enforce physical range and consistency constraints"""
        try:
            p_coeff = predictions[:, 0]
            tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
            
            # 1. Pressure coefficient should be reasonable (typically -2 to 2 for most flows)
            p_range_penalty = torch.mean(F.relu(torch.abs(p_coeff) - 3.0))  # Penalty for |Cp| > 3
            
            # 2. Shear stress components should not be extreme
            tau_magnitude = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-8)
            tau_range_penalty = torch.mean(F.relu(tau_magnitude - 10.0))  # Penalty for extreme shear
            
            # 3. Pressure and shear correlation (they should be related in physical flows)
            if len(p_coeff) > 1 and len(tau_magnitude) > 1:
                correlation_loss = 1.0 - torch.abs(torch.corrcoef(torch.stack([p_coeff, tau_magnitude]))[0, 1])
            else:
                correlation_loss = torch.tensor(0.0, device=predictions.device)
            
            return {
                'physical_consistency_loss': p_range_penalty + tau_range_penalty + 0.1 * correlation_loss,
                'pressure_range_penalty': p_range_penalty,
                'shear_range_penalty': tau_range_penalty,
                'pressure_shear_correlation_loss': correlation_loss,
                'pressure_coefficient_range': (torch.min(p_coeff), torch.max(p_coeff)),
                'shear_magnitude_range': (torch.min(tau_magnitude), torch.max(tau_magnitude))
            }
            
        except Exception as e:
            print(f"Warning: Physical consistency loss failed: {e}")
            return {
                'physical_consistency_loss': torch.tensor(0.0, device=predictions.device),
                'pressure_range_penalty': torch.tensor(0.0, device=predictions.device),
                'shear_range_penalty': torch.tensor(0.0, device=predictions.device),
                'pressure_shear_correlation_loss': torch.tensor(0.0, device=predictions.device),
                'pressure_coefficient_range': (torch.tensor(0.0), torch.tensor(0.0)),
                'shear_magnitude_range': (torch.tensor(0.0), torch.tensor(0.0))
            }
    
    def spatial_coherence_loss(self, predictions: torch.Tensor, data: Any) -> Dict[str, torch.Tensor]:
        """Enforce spatial coherence in fields"""
        try:
            if hasattr(data, 'edge_index'):
                edge_index = data.edge_index
                row, col = edge_index
                
                # Compute differences across edges
                edge_diff = predictions[row] - predictions[col]
                
                # Separate losses for each component
                p_coherence = torch.mean(edge_diff[:, 0]**2)
                tau_x_coherence = torch.mean(edge_diff[:, 1]**2)
                tau_y_coherence = torch.mean(edge_diff[:, 2]**2)
                tau_z_coherence = torch.mean(edge_diff[:, 3]**2)
                
                total_coherence = p_coherence + tau_x_coherence + tau_y_coherence + tau_z_coherence
                
                return {
                    'spatial_coherence_loss': total_coherence,
                    'pressure_coherence': p_coherence,
                    'shear_x_coherence': tau_x_coherence,
                    'shear_y_coherence': tau_y_coherence,
                    'shear_z_coherence': tau_z_coherence
                }
            else:
                return {
                    'spatial_coherence_loss': torch.tensor(0.0, device=predictions.device),
                    'pressure_coherence': torch.tensor(0.0, device=predictions.device),
                    'shear_x_coherence': torch.tensor(0.0, device=predictions.device),
                    'shear_y_coherence': torch.tensor(0.0, device=predictions.device),
                    'shear_z_coherence': torch.tensor(0.0, device=predictions.device)
                }
                
        except Exception as e:
            print(f"Warning: Spatial coherence loss failed: {e}")
            return {
                'spatial_coherence_loss': torch.tensor(0.0, device=predictions.device),
                'pressure_coherence': torch.tensor(0.0, device=predictions.device),
                'shear_x_coherence': torch.tensor(0.0, device=predictions.device),
                'shear_y_coherence': torch.tensor(0.0, device=predictions.device),
                'shear_z_coherence': torch.tensor(0.0, device=predictions.device)
            }


def compute_corrected_comprehensive_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                                       data: Any, loss_weights: Optional[Dict[str, float]] = None,
                                       return_components: bool = True) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive loss for pressure coefficient and wall shear stress data
    
    Args:
        predictions: [N, 4] tensor [pressure_coeff, tau_x, tau_y, tau_z]
        targets: [N, 4] tensor with same structure
        data: Data object containing position and connectivity info
        loss_weights: Optional custom loss weights
        return_components: Whether to return detailed component breakdown
    
    Returns:
        Dictionary with loss components and total loss
    """
    
    device = predictions.device
    pinn_loss = CorrectedPINNLoss(loss_weights)
    pinn_loss = pinn_loss.to(device)
    
    loss_components = {}
    
    # 1. Basic MSE Loss (most important for data fitting)
    mse_loss = F.mse_loss(predictions, targets)
    loss_components['mse'] = mse_loss
    
    # 2. Pressure-based losses
    try:
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos.clone().detach().requires_grad_(True)
            pred_physics = predictions.clone().requires_grad_(True)
            
            # Pressure smoothness
            pressure_results = pinn_loss.pressure_smoothness_loss(pred_physics, pos)
            loss_components.update(pressure_results)
            
        else:
            # Fallback without spatial gradients
            p_coeff = predictions[:, 0]
            if len(p_coeff) > 1:
                pressure_smoothness = torch.var(torch.diff(p_coeff))
                loss_components['pressure_smoothness_loss'] = pressure_smoothness
            else:
                loss_components['pressure_smoothness_loss'] = torch.tensor(0.0, device=device)
                
    except Exception as e:
        print(f"Warning: Pressure loss computation failed: {e}")
        loss_components['pressure_smoothness_loss'] = torch.tensor(0.0, device=device)
    
    # 3. Wall shear stress losses
    try:
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos.clone().detach().requires_grad_(True)
            pred_physics = predictions.clone().requires_grad_(True)
            
            shear_results = pinn_loss.shear_stress_losses(pred_physics, pos)
            loss_components.update(shear_results)
        else:
            # Simplified shear losses without gradients
            tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
            if len(tau_x) > 1:
                shear_smoothness = (torch.var(torch.diff(tau_x)) + 
                                  torch.var(torch.diff(tau_y)) + 
                                  torch.var(torch.diff(tau_z))) / 3
                loss_components['shear_stress_smoothness_loss'] = shear_smoothness
            else:
                loss_components['shear_stress_smoothness_loss'] = torch.tensor(0.0, device=device)
                
    except Exception as e:
        print(f"Warning: Shear stress loss computation failed: {e}")
        loss_components['shear_stress_smoothness_loss'] = torch.tensor(0.0, device=device)
    
    # 4. Wall shear balance loss
    try:
        balance_results = pinn_loss.wall_shear_balance_loss(predictions)
        loss_components.update(balance_results)
    except Exception as e:
        print(f"Warning: Wall shear balance loss failed: {e}")
        loss_components['wall_shear_balance_loss'] = torch.tensor(0.0, device=device)
    
    # 5. Physical consistency
    try:
        consistency_results = pinn_loss.physical_consistency_loss(predictions)
        loss_components.update(consistency_results)
    except Exception as e:
        print(f"Warning: Physical consistency loss failed: {e}")
        loss_components['physical_consistency_loss'] = torch.tensor(0.0, device=device)
    
    # 6. Spatial coherence
    try:
        coherence_results = pinn_loss.spatial_coherence_loss(predictions, data)
        loss_components.update(coherence_results)
    except Exception as e:
        print(f"Warning: Spatial coherence loss failed: {e}")
        loss_components['spatial_coherence_loss'] = torch.tensor(0.0, device=device)
    
    # 7. Simple gradient-based losses for available data
    if len(predictions) > 1:
        # Pressure gradient loss
        p_grad = torch.diff(predictions[:, 0])
        loss_components['pressure_gradient'] = torch.mean(p_grad**2) * 0.1
        
        # Multi-component correlation
        tau_magnitude = torch.sqrt(predictions[:, 1]**2 + predictions[:, 2]**2 + predictions[:, 3]**2 + 1e-8)
        if len(tau_magnitude) > 1:
            corr_matrix = torch.corrcoef(torch.stack([predictions[:, 0], tau_magnitude]))
            correlation_loss = 1.0 - torch.abs(corr_matrix[0, 1])
            loss_components['multi_component_correlation'] = correlation_loss
        else:
            loss_components['multi_component_correlation'] = torch.tensor(0.0, device=device)
    else:
        loss_components['pressure_gradient'] = torch.tensor(0.0, device=device)
        loss_components['multi_component_correlation'] = torch.tensor(0.0, device=device)
    
    # 8. Compute weighted total loss
    weights = loss_weights or pinn_loss.default_weights
    total_loss = torch.tensor(0.0, device=device)
    
    for loss_name, loss_value in loss_components.items():
        if loss_name in weights and isinstance(loss_value, torch.Tensor):
            weighted_loss = weights[loss_name] * loss_value
            total_loss = total_loss + weighted_loss
            loss_components[f'{loss_name}_weighted'] = weighted_loss
    
    loss_components['total_loss'] = total_loss
    
    # 9. Add statistics
    valid_losses = [v for k, v in loss_components.items() 
                   if isinstance(v, torch.Tensor) and not k.endswith('_weighted') 
                   and k not in ['total_loss', 'pressure_coefficient_range', 'shear_magnitude_range']]
    
    if valid_losses:
        loss_components['loss_statistics'] = {
            'mean': torch.stack(valid_losses).mean(),
            'std': torch.stack(valid_losses).std(),
            'max': torch.stack(valid_losses).max(),
            'min': torch.stack(valid_losses).min(),
            'num_components': len(valid_losses)
        }
    
    if return_components:
        return {
            'loss_components': loss_components,
            'total_loss': total_loss,
            'weights_used': weights,
            'data_structure': 'pressure_coefficient + wall_shear_stress_components'
        }
    else:
        return total_loss