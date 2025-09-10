"""
Progressive Physics Loss Integration
Step-by-step approach to add physics constraints
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

class ProgressivePhysicsLoss:
    """Progressive loss that adds physics terms step by step"""
    
    def __init__(self):
        self.current_level = 1  # Start with MSE only
        self.max_level = 5      # Maximum complexity
        
    def set_level(self, level: int):
        """Set current physics loss level"""
        self.current_level = min(max(1, level), self.max_level)
        print(f"📊 Physics loss level set to: {self.current_level}")
        
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                    data: Any) -> Dict[str, torch.Tensor]:
        """Compute loss based on current level"""
        
        if self.current_level == 1:
            return self._level_1_mse_only(predictions, targets)
        elif self.current_level == 2:
            return self._level_2_add_smoothness(predictions, targets)
        elif self.current_level == 3:
            return self._level_3_add_physical_constraints(predictions, targets)
        elif self.current_level == 4:
            return self._level_4_add_spatial_coherence(predictions, targets, data)
        elif self.current_level == 5:
            return self._level_5_full_physics(predictions, targets, data)
        else:
            return self._level_1_mse_only(predictions, targets)
    
    def _level_1_mse_only(self, predictions, targets):
        """Level 1: MSE 손실만 사용"""
        print("🔵 Level 1: MSE Only")
        
        mse_loss = F.mse_loss(predictions, targets)
        
        return {
            'mse': mse_loss,
            'total_loss': mse_loss,
            'level': 1
        }
    
    def _level_2_add_smoothness(self, predictions, targets):
        """Level 2: MSE + 평활도 손실"""
        print("🟡 Level 2: MSE + Smoothness")
        
        device = predictions.device
        mse_loss = F.mse_loss(predictions, targets)
        
        # Simple smoothness loss
        smoothness_loss = torch.tensor(0.0, device=device)
        if len(predictions) > 1:
            for i in range(4):  # For each output component
                diff = torch.diff(predictions[:, i])
                smoothness_loss += torch.mean(diff**2)
            smoothness_loss = smoothness_loss / 4
        
        total_loss = mse_loss + 0.1 * smoothness_loss
        
        return {
            'mse': mse_loss,
            'smoothness': smoothness_loss,
            'total_loss': total_loss,
            'level': 2
        }
    
    def _level_3_add_physical_constraints(self, predictions, targets):
        """Level 3: MSE + 평활도 + 물리적 범위 제약"""
        print("🟠 Level 3: MSE + Smoothness + Physical Constraints")
        
        device = predictions.device
        
        # Previous level losses
        prev_result = self._level_2_add_smoothness(predictions, targets)
        mse_loss = prev_result['mse']
        smoothness_loss = prev_result['smoothness']
        
        # Physical constraint losses
        p_coeff = predictions[:, 0]
        tau_x, tau_y, tau_z = predictions[:, 1], predictions[:, 2], predictions[:, 3]
        
        # 1. Pressure coefficient range penalty
        pressure_penalty = F.relu(torch.abs(p_coeff) - 3.0).mean()
        
        # 2. Shear stress magnitude penalty
        tau_magnitude = torch.sqrt(tau_x**2 + tau_y**2 + tau_z**2 + 1e-8)
        shear_penalty = F.relu(tau_magnitude - 5.0).mean()
        
        # 3. Component balance (no single component should dominate extremely)
        tau_components = torch.stack([torch.abs(tau_x), torch.abs(tau_y), torch.abs(tau_z)], dim=1)
        max_component = torch.max(tau_components, dim=1)[0]
        mean_component = torch.mean(tau_components, dim=1)
        balance_penalty = F.relu(max_component - 3 * mean_component).mean()
        
        physical_loss = pressure_penalty + shear_penalty + balance_penalty
        total_loss = mse_loss + 0.1 * smoothness_loss + 0.05 * physical_loss
        
        return {
            'mse': mse_loss,
            'smoothness': smoothness_loss,
            'pressure_penalty': pressure_penalty,
            'shear_penalty': shear_penalty,
            'balance_penalty': balance_penalty,
            'physical_constraints': physical_loss,
            'total_loss': total_loss,
            'level': 3
        }
    
    def _level_4_add_spatial_coherence(self, predictions, targets, data):
        """Level 4: Level 3 + 공간적 일관성"""
        print("🔴 Level 4: Add Spatial Coherence")
        
        device = predictions.device
        
        # Previous level losses
        prev_result = self._level_3_add_physical_constraints(predictions, targets)
        total_prev = prev_result['total_loss']
        
        # Spatial coherence loss
        spatial_loss = torch.tensor(0.0, device=device)
        
        if hasattr(data, 'edge_index'):
            row, col = data.edge_index
            
            # Compute differences across edges
            edge_diff = predictions[row] - predictions[col]
            
            # Penalize large differences for each component
            for i in range(4):
                spatial_loss += torch.mean(edge_diff[:, i]**2)
            spatial_loss = spatial_loss / 4
        
        total_loss = total_prev + 0.05 * spatial_loss
        
        result = prev_result.copy()
        result.update({
            'spatial_coherence': spatial_loss,
            'total_loss': total_loss,
            'level': 4
        })
        
        return result
    
    def _level_5_full_physics(self, predictions, targets, data):
        """Level 5: 모든 물리 제약 (그래디언트 기반)"""
        print("🟣 Level 5: Full Physics with Gradients")
        
        device = predictions.device
        
        # Previous level losses
        prev_result = self._level_4_add_spatial_coherence(predictions, targets, data)
        total_prev = prev_result['total_loss']
        
        # Advanced physics losses (if possible)
        gradient_loss = torch.tensor(0.0, device=device)
        
        try:
            if hasattr(data, 'pos') and data.pos is not None and predictions.requires_grad:
                # Simple gradient-based loss
                p_coeff = predictions[:, 0:1]
                
                # Try to compute gradients
                try:
                    dp_dx = torch.autograd.grad(
                        p_coeff.sum(), data.pos, 
                        create_graph=True, retain_graph=True, allow_unused=True
                    )[0]
                    
                    if dp_dx is not None:
                        # Gradient magnitude penalty (prevent extreme gradients)
                        grad_magnitude = torch.norm(dp_dx, dim=1)
                        gradient_loss = F.relu(grad_magnitude - 2.0).mean()
                
                except Exception as e:
                    print(f"Gradient computation failed: {e}")
                    
        except Exception as e:
            print(f"Advanced physics loss failed: {e}")
        
        total_loss = total_prev + 0.02 * gradient_loss
        
        result = prev_result.copy()
        result.update({
            'gradient_penalty': gradient_loss,
            'total_loss': total_loss,
            'level': 5
        })
        
        return result


def create_progressive_training_schedule():
    """Create training schedule for progressive physics loss"""
    
    schedule = {
        # Epoch ranges and corresponding loss levels
        'epochs_0_20': 1,    # Epochs 0-19: MSE only
        'epochs_20_40': 2,   # Epochs 20-39: MSE + Smoothness
        'epochs_40_60': 3,   # Epochs 40-59: + Physical constraints
        'epochs_60_80': 4,   # Epochs 60-79: + Spatial coherence
        'epochs_80_100': 5   # Epochs 80-99: Full physics
    }
    
    return schedule

def get_loss_level_for_epoch(epoch: int) -> int:
    """Get appropriate loss level for current epoch"""
    
    if epoch < 20:
        return 1
    elif epoch < 40:
        return 2
    elif epoch < 60:
        return 3
    elif epoch < 80:
        return 4
    else:
        return 5

def progressive_train_epoch(model, train_loader, optimizer, scheduler, epoch, config):
    """Training function with progressive physics loss"""
    
    # Initialize progressive loss
    physics_loss = ProgressivePhysicsLoss()
    
    # Set loss level based on epoch
    level = get_loss_level_for_epoch(epoch)
    physics_loss.set_level(level)
    
    model.train()
    total_loss = 0
    loss_components_sum = {}
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} (Level {level})')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Compute progressive loss
            loss_result = physics_loss.compute_loss(predictions, batch.y, batch)
            
            loss = loss_result['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Track losses
            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_result.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0
                    loss_components_sum[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'level': level,
                'mse': loss_result.get('mse', torch.tensor(0)).item()
            })
                
        except Exception as e:
            print(f"\\nError in batch {batch_idx}: {e}")
            continue
    
    # Average losses
    avg_loss = total_loss / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


if __name__ == "__main__":
    print("🚀 Progressive Physics Loss System")
    print("\\n📋 Training Schedule:")
    print("  Epochs 0-19:  Level 1 - MSE only")
    print("  Epochs 20-39: Level 2 - MSE + Smoothness")
    print("  Epochs 40-59: Level 3 - + Physical constraints")
    print("  Epochs 60-79: Level 4 - + Spatial coherence")
    print("  Epochs 80-99: Level 5 - Full physics")
    print("\\n✅ This approach prevents training instability!")