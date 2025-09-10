"""
Enhanced Validation with Relative L2 Error per Component
Provides detailed error analysis for each output component
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_relative_l2_errors(predictions, targets):
    """
    Compute relative L2 error for each component
    
    Args:
        predictions: [N, 4] tensor [pressure_coeff, tau_x, tau_y, tau_z]
        targets: [N, 4] tensor with same structure
        
    Returns:
        dict: Relative L2 errors and additional metrics per component
    """
    
    component_names = ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']
    metrics = {}
    
    for i, comp_name in enumerate(component_names):
        pred_comp = predictions[:, i]
        target_comp = targets[:, i]
        
        # L2 norm of prediction error
        error_l2 = torch.norm(pred_comp - target_comp, p=2)
        
        # L2 norm of target (for normalization)
        target_l2 = torch.norm(target_comp, p=2)
        
        # Relative L2 error
        if target_l2 > 1e-10:  # Avoid division by zero
            relative_l2 = error_l2 / target_l2
        else:
            relative_l2 = error_l2  # If target is near zero, use absolute error
        
        # Additional component-wise metrics
        mse_comp = F.mse_loss(pred_comp, target_comp)
        mae_comp = F.l1_loss(pred_comp, target_comp)
        
        # Max absolute error
        max_abs_error = torch.max(torch.abs(pred_comp - target_comp))
        
        # R² coefficient (coefficient of determination)
        target_mean = torch.mean(target_comp)
        ss_tot = torch.sum((target_comp - target_mean) ** 2)
        ss_res = torch.sum((target_comp - pred_comp) ** 2)
        
        if ss_tot > 1e-10:
            r2_score = 1 - (ss_res / ss_tot)
        else:
            r2_score = torch.tensor(0.0, device=predictions.device)
        
        # Store metrics
        metrics[f'{comp_name}_relative_l2_error'] = relative_l2
        metrics[f'{comp_name}_mse'] = mse_comp
        metrics[f'{comp_name}_mae'] = mae_comp  
        metrics[f'{comp_name}_max_abs_error'] = max_abs_error
        metrics[f'{comp_name}_r2_score'] = r2_score
        
        # Statistical info
        metrics[f'{comp_name}_pred_mean'] = torch.mean(pred_comp)
        metrics[f'{comp_name}_pred_std'] = torch.std(pred_comp)
        metrics[f'{comp_name}_target_mean'] = torch.mean(target_comp)
        metrics[f'{comp_name}_target_std'] = torch.std(target_comp)
    
    # Overall metrics
    overall_relative_l2 = torch.norm(predictions - targets, p=2) / torch.norm(targets, p=2)
    metrics['overall_relative_l2_error'] = overall_relative_l2
    metrics['overall_mse'] = F.mse_loss(predictions, targets)
    metrics['overall_mae'] = F.l1_loss(predictions, targets)
    
    return metrics

def enhanced_validate_epoch(model, val_loader, epoch, config, device):
    """
    Enhanced validation with detailed component-wise analysis
    """
    
    from progressive_loss import ProgressivePhysicsLoss, get_loss_level_for_epoch
    
    physics_loss = ProgressivePhysicsLoss()
    level = get_loss_level_for_epoch(epoch)
    physics_loss.set_level(level)
    
    model.eval()
    total_loss = 0
    loss_components_sum = {}
    
    # For component-wise error accumulation
    all_predictions = []
    all_targets = []
    
    num_batches = len(val_loader)
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                batch = batch.to(device)
                
                # Forward pass
                predictions = model(batch)
                
                # Accumulate predictions and targets for component analysis
                all_predictions.append(predictions)
                all_targets.append(batch.y)
                
                # Compute loss
                loss_result = physics_loss.compute_loss(predictions, batch.y, batch)
                
                loss = loss_result['total_loss']
                total_loss += loss.item()
                successful_batches += 1
                
                # Accumulate loss components
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
    
    # Component-wise analysis
    if all_predictions:
        # Concatenate all predictions and targets
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        
        # Compute detailed metrics
        component_metrics = compute_relative_l2_errors(all_pred, all_targ)
        
        # Merge with loss components
        avg_components.update(component_metrics)
        
        # Print component-wise results
        print(f"\\n📊 Component-wise Validation Results (Epoch {epoch+1}):")
        print(f"{'Component':<15} {'Rel L2 Error':<12} {'MSE':<10} {'R²':<8} {'Max Abs Err':<12}")
        print("-" * 65)
        
        for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
            rel_l2 = component_metrics[f'{comp}_relative_l2_error'].item()
            mse = component_metrics[f'{comp}_mse'].item()
            r2 = component_metrics[f'{comp}_r2_score'].item()
            max_err = component_metrics[f'{comp}_max_abs_error'].item()
            
            print(f"{comp:<15} {rel_l2:<12.6f} {mse:<10.6f} {r2:<8.4f} {max_err:<12.6f}")
        
        # Overall metrics
        overall_rel_l2 = component_metrics['overall_relative_l2_error'].item()
        overall_mse = component_metrics['overall_mse'].item()
        print(f"{'OVERALL':<15} {overall_rel_l2:<12.6f} {overall_mse:<10.6f}")
    
    # Log to WandB with component details
    try:
        import wandb
        
        wandb_log = {
            'val/loss': avg_loss,
            'epoch': epoch,
            'val/physics_level': level
        }
        
        # Add loss components
        component_items = [(k, v) for k, v in avg_components.items()]
        for key, value in component_items:
            if isinstance(value, torch.Tensor):
                wandb_log[f'val/{key}'] = value.item()
            else:
                wandb_log[f'val/{key}'] = value
        
        # Specific component error logging
        if all_predictions:
            # Main metrics for dashboard
            for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
                wandb_log[f'val/errors/{comp}_relative_l2'] = component_metrics[f'{comp}_relative_l2_error'].item()
                wandb_log[f'val/errors/{comp}_r2_score'] = component_metrics[f'{comp}_r2_score'].item()
                wandb_log[f'val/stats/{comp}_mse'] = component_metrics[f'{comp}_mse'].item()
                wandb_log[f'val/stats/{comp}_mae'] = component_metrics[f'{comp}_mae'].item()
            
            # Overall metrics
            wandb_log['val/errors/overall_relative_l2'] = component_metrics['overall_relative_l2_error'].item()
            wandb_log['val/errors/overall_mse'] = component_metrics['overall_mse'].item()
        
        wandb.log(wandb_log)
        
    except Exception as e:
        print(f"WandB logging failed: {e}")
    
    return avg_loss, avg_components

def enhanced_train_epoch_with_metrics(model, train_loader, optimizer, scheduler, epoch, config, device):
    """
    Enhanced training epoch that also computes component metrics periodically
    """
    
    from progressive_loss import ProgressivePhysicsLoss, get_loss_level_for_epoch
    
    physics_loss = ProgressivePhysicsLoss()
    level = get_loss_level_for_epoch(epoch)
    physics_loss.set_level(level)
    
    model.train()
    total_loss = 0
    loss_components_sum = {}
    num_batches = len(train_loader)
    successful_batches = 0
    
    # For periodic component analysis during training
    train_predictions = []
    train_targets = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} (Level {level})')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            predictions = model(batch)
            if not predictions.requires_grad:
                predictions = predictions.requires_grad_(True)
            
            # Store for component analysis (every 10 batches to save memory)
            if batch_idx % 10 == 0:
                train_predictions.append(predictions.detach())
                train_targets.append(batch.y)
            
            loss_result = physics_loss.compute_loss(predictions, batch.y, batch)
            loss = loss_result['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            successful_batches += 1
            
            # Accumulate loss components
            loss_items = [(k, v) for k, v in loss_result.items() 
                         if isinstance(v, torch.Tensor) and v.dim() == 0]
            
            for key, value in loss_items:
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0
                loss_components_sum[key] += value.item()
            
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
        if not hasattr(scheduler, 'mode'):
            scheduler.step()
    
    avg_loss = total_loss / max(1, successful_batches)
    avg_components = {k: v / max(1, successful_batches) for k, v in loss_components_sum.items()}
    
    # Periodic component analysis during training
    if train_predictions and epoch % 5 == 0:  # Every 5 epochs
        all_train_pred = torch.cat(train_predictions, dim=0)
        all_train_targ = torch.cat(train_targets, dim=0)
        train_component_metrics = compute_relative_l2_errors(all_train_pred, all_train_targ)
        
        # Log training component metrics
        try:
            import wandb
            train_wandb_log = {
                'train/loss': avg_loss,
                'epoch': epoch
            }
            
            for comp in ['pressure_coeff', 'tau_x', 'tau_y', 'tau_z']:
                train_wandb_log[f'train/errors/{comp}_relative_l2'] = train_component_metrics[f'{comp}_relative_l2_error'].item()
            
            wandb.log(train_wandb_log)
        except:
            pass
    
    return avg_loss, avg_components

print("✅ Enhanced validation with component-wise metrics loaded!")
print("\\n📊 Features:")
print("  - Relative L2 error per component")
print("  - R² score per component") 
print("  - MSE, MAE, Max absolute error per component")
print("  - Detailed WandB logging")
print("  - Statistical analysis")
print("\\n🎯 Use enhanced_validate_epoch() instead of regular validation!")