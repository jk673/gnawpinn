def train_epoch(model, train_loader, optimizer, scheduler, epoch, config):
    """Train for one epoch with comprehensive loss logging - Fixed"""
    model.train()
    total_loss = 0
    loss_components_sum = {}
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Compute corrected comprehensive loss
            loss_result = compute_corrected_comprehensive_loss(
                predictions, batch.y, batch, 
                loss_weights=config['loss_weights'],
                return_components=True
            )
            
            loss = loss_result['total_loss']
            loss_components = loss_result['loss_components']
            
            # Gradient accumulation
            loss = loss / config['training']['accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['training']['accumulation_steps'] == 0:
                # Gradient clipping
                if config['training']['gradient_clip_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Track losses
            actual_loss = loss.item() * config['training']['accumulation_steps']
            total_loss += actual_loss
            
            # Accumulate loss components - FIXED: Create list first to avoid dict modification during iteration
            loss_items = list(loss_components.items())  # Create a copy of items
            for key, value in loss_items:
                if isinstance(value, torch.Tensor) and value.dim() == 0:  # scalar tensor
                    if key not in loss_components_sum:
                        loss_components_sum[key] = 0
                    loss_components_sum[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': actual_loss,
                'mse': loss_components.get('mse', torch.tensor(0)).item(),
                'p_smooth': loss_components.get('pressure_smoothness_loss', torch.tensor(0)).item()
            })
            
            # WandB logging
            if batch_idx % config['wandb']['log_freq'] == 0:
                step = epoch * num_batches + batch_idx
                
                # Basic metrics
                wandb_log = {
                    'train/batch_loss': actual_loss,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': step
                }
                
                # Corrected loss components - FIXED: Create list first
                loss_items_wandb = list(loss_components.items())
                for key, value in loss_items_wandb:
                    if isinstance(value, torch.Tensor) and value.dim() == 0:
                        wandb_log[f'train/loss_components/{key}'] = value.item()
                
                # Memory usage
                if batch_idx % config['memory']['memory_check_freq'] == 0:
                    memory_usage = get_memory_usage()
                    wandb_log.update({
                        'system/cpu_memory_gb': memory_usage['cpu_gb'],
                        'system/gpu_memory_gb': memory_usage['gpu_gb']
                    })
                
                # Gradient norms
                if config['wandb']['log_gradients']:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    wandb_log['train/gradient_norm'] = total_norm
                
                wandb.log(wandb_log)
                
        except Exception as e:
            print(f"\\nError in batch {batch_idx}: {e}")
            continue
    
    # Step scheduler
    if scheduler and scheduler_type != 'plateau':
        scheduler.step()
    
    # Average losses
    avg_loss = total_loss / max(1, num_batches)  # Avoid division by zero
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components_sum.items()}
    
    return avg_loss, avg_components


def validate_epoch(model, val_loader, epoch, config):
    """Validate for one epoch - Fixed"""
    model.eval()
    total_loss = 0
    loss_components_sum = {}
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                batch = batch.to(device)
                predictions = model(batch)
                
                # Compute corrected comprehensive loss
                loss_result = compute_corrected_comprehensive_loss(
                    predictions, batch.y, batch,
                    loss_weights=config['loss_weights'],
                    return_components=True
                )
                
                loss = loss_result['total_loss']
                loss_components = loss_result['loss_components']
                
                total_loss += loss.item()
                
                # Accumulate components - FIXED: Create list first
                loss_items = list(loss_components.items())
                for key, value in loss_items:
                    if isinstance(value, torch.Tensor) and value.dim() == 0:
                        if key not in loss_components_sum:
                            loss_components_sum[key] = 0
                        loss_components_sum[key] += value.item()
                        
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    avg_loss = total_loss / max(1, num_batches)  # Avoid division by zero
    avg_components = {k: v / max(1, num_batches) for k, v in loss_components_sum.items()}
    
    # Log validation metrics - FIXED: Create list first
    wandb_log = {
        'val/loss': avg_loss,
        'epoch': epoch
    }
    
    component_items = list(avg_components.items())
    for key, value in component_items:
        wandb_log[f'val/loss_components/{key}'] = value
    
    wandb.log(wandb_log)
    
    return avg_loss, avg_components


print("Fixed training functions defined for CFDSurrogateModel")