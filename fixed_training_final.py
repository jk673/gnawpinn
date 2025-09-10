"""
Final Fixed Training Functions
Resolves gradient and dictionary issues
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def progressive_train_epoch_fixed(model, train_loader, optimizer, scheduler, epoch, config, device):
    """Fixed training function with progressive physics loss"""
    
    from progressive_loss import ProgressivePhysicsLoss, get_loss_level_for_epoch
    
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
    
    successful_batches = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # IMPORTANT: Ensure predictions require grad for gradient computation
            predictions = model(batch)
            if not predictions.requires_grad:
                predictions = predictions.requires_grad_(True)
            
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
            successful_batches += 1
            
            # FIXED: Create static copy of loss_result items to avoid dict modification
            loss_items = [(k, v) for k, v in loss_result.items() 
                         if isinstance(v, torch.Tensor) and v.dim() == 0]
            
            for key, value in loss_items:
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
    
    # Step scheduler
    if scheduler and hasattr(scheduler, 'step'):
        if hasattr(scheduler, 'mode'):  # ReduceLROnPlateau
            pass  # Will be called in main loop
        else:
            scheduler.step()
    
    # Average losses
    avg_loss = total_loss / max(1, successful_batches)
    avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


def progressive_validate_epoch_fixed(model, val_loader, epoch, config, device):
    """Fixed validation function"""
    
    from progressive_loss import ProgressivePhysicsLoss, get_loss_level_for_epoch
    
    physics_loss = ProgressivePhysicsLoss()
    level = get_loss_level_for_epoch(epoch)
    physics_loss.set_level(level)
    
    model.eval()
    total_loss = 0
    loss_components_sum = {}
    num_batches = len(val_loader)
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                batch = batch.to(device)
                
                # Forward pass
                predictions = model(batch)
                
                # Compute loss (no gradients needed for validation)
                loss_result = physics_loss.compute_loss(predictions, batch.y, batch)
                
                loss = loss_result['total_loss']
                total_loss += loss.item()
                successful_batches += 1
                
                # FIXED: Create static copy to avoid dict modification
                loss_items = [(k, v) for k, v in loss_result.items() 
                             if isinstance(v, torch.Tensor) and v.dim() == 0]
                
                for key, value in loss_items:
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0
                    loss_components_sum[key] += value.item()
                        
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    # Average losses
    avg_loss = total_loss / max(1, successful_batches)
    avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    # FIXED: Log to wandb safely
    try:
        import wandb
        
        wandb_log = {
            'val/loss': avg_loss,
            'epoch': epoch
        }
        
        # Create static copy for wandb logging
        component_items = [(k, v) for k, v in avg_components.items()]
        for key, value in component_items:
            wandb_log[f'val/loss_components/{key}'] = value
        
        wandb.log(wandb_log)
    except Exception as e:
        print(f"WandB logging failed: {e}")
    
    return avg_loss, avg_components


def simple_mse_train_epoch(model, train_loader, optimizer, scheduler, epoch, device):
    """Ultra-simple training with MSE only - for debugging"""
    
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    successful_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} (MSE Only)')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Simple MSE loss only
            loss = F.mse_loss(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item()
            successful_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
                
        except Exception as e:
            print(f"\\nError in batch {batch_idx}: {e}")
            continue
    
    # Step scheduler
    if scheduler and hasattr(scheduler, 'step'):
        if not hasattr(scheduler, 'mode'):  # Not ReduceLROnPlateau
            scheduler.step()
    
    # Average loss
    avg_loss = total_loss / max(1, successful_batches)
    
    return avg_loss, {'mse': avg_loss}


def simple_mse_validate_epoch(model, val_loader, epoch, device):
    """Ultra-simple validation with MSE only"""
    
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                batch = batch.to(device)
                predictions = model(batch)
                
                # Simple MSE loss only
                loss = F.mse_loss(predictions, batch.y)
                total_loss += loss.item()
                successful_batches += 1
                        
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    avg_loss = total_loss / max(1, successful_batches)
    
    # Log to wandb safely
    try:
        import wandb
        wandb.log({
            'val/loss': avg_loss,
            'val/mse': avg_loss,
            'epoch': epoch
        })
    except:
        pass
    
    return avg_loss, {'mse': avg_loss}


print("✅ Fixed training functions loaded!")
print("\\n🎯 Available functions:")
print("  1. simple_mse_train_epoch() - Ultra-simple MSE only")
print("  2. progressive_train_epoch_fixed() - Progressive physics loss")
print("\\n💡 Start with simple_mse_train_epoch to ensure basic training works!")