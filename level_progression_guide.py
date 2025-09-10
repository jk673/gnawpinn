"""
Level Progression Guide - How to advance to next physics loss level
"""

def check_training_readiness(train_losses, val_losses, min_epochs=5):
    """
    Check if model is ready to advance to next level
    
    Args:
        train_losses: List of recent training losses
        val_losses: List of recent validation losses  
        min_epochs: Minimum epochs to train at current level
        
    Returns:
        bool: True if ready to advance
    """
    
    if len(train_losses) < min_epochs:
        print(f"⏳ Need at least {min_epochs} epochs at current level")
        return False
    
    # Check if loss is stabilizing (not decreasing rapidly)
    recent_train = train_losses[-5:]
    recent_val = val_losses[-5:]
    
    # Calculate loss change rate
    if len(recent_train) >= 2:
        train_change = (recent_train[-1] - recent_train[0]) / recent_train[0]
        val_change = (recent_val[-1] - recent_val[0]) / recent_val[0]
        
        print(f"📊 Recent loss changes:")
        print(f"  Train: {train_change*100:.2f}%")
        print(f"  Val: {val_change*100:.2f}%")
        
        # Ready to advance if:
        # 1. Training loss change is small (< 5% improvement in last 5 epochs)
        # 2. Validation loss is not increasing rapidly
        if abs(train_change) < 0.05 and val_change < 0.1:
            print("✅ Loss has stabilized - ready to advance!")
            return True
        else:
            print("⏳ Loss still improving - continue current level")
            return False
    
    return False

def advance_to_next_level(current_level, max_level=5):
    """Advance to next physics loss level"""
    
    if current_level >= max_level:
        print(f"🎯 Already at maximum level ({max_level})")
        return current_level
    
    next_level = current_level + 1
    
    level_descriptions = {
        2: "🟡 Level 2: Adding Smoothness Loss",
        3: "🟠 Level 3: Adding Physical Constraints", 
        4: "🔴 Level 4: Adding Spatial Coherence",
        5: "🟣 Level 5: Full Physics with Gradients"
    }
    
    print(f"\n🚀 ADVANCING TO NEXT LEVEL!")
    print(f"From Level {current_level} → Level {next_level}")
    print(f"{level_descriptions.get(next_level, 'Unknown level')}")
    
    return next_level

def manual_level_control():
    """Manual control for level progression"""
    
    print("""
🎮 MANUAL LEVEL CONTROL:

To manually advance to next level, run:

```python
# 1. Check current progress
current_epoch = epoch  # Your current epoch
current_level = get_loss_level_for_epoch(current_epoch)
print(f"Current Level: {current_level}")

# 2. Check if ready to advance  
ready = check_training_readiness(train_losses[-10:], val_losses[-10:])

# 3. Force advance to next level (override epoch-based scheduling)
if ready:
    # Method 1: Modify the epoch-based function
    def get_loss_level_for_epoch_override(epoch):
        return min(current_level + 1, 5)  # Force next level
    
    # Method 2: Directly set level in progressive loss
    physics_loss = ProgressivePhysicsLoss()
    physics_loss.set_level(current_level + 1)
```
""")

def adaptive_level_progression(train_losses, val_losses, current_epoch, 
                              force_advance=False):
    """
    Adaptive level progression based on training performance
    
    Args:
        train_losses: Recent training losses
        val_losses: Recent validation losses  
        current_epoch: Current epoch number
        force_advance: Force advance regardless of performance
        
    Returns:
        int: Recommended level
    """
    
    # Default epoch-based level
    from progressive_loss import get_loss_level_for_epoch
    default_level = get_loss_level_for_epoch(current_epoch)
    
    if force_advance:
        recommended_level = min(default_level + 1, 5)
        print(f"🚨 FORCE ADVANCE: Level {default_level} → {recommended_level}")
        return recommended_level
    
    # Check if ready to advance beyond default
    if len(train_losses) >= 5:
        ready = check_training_readiness(train_losses, val_losses)
        
        if ready and current_epoch >= 15:  # Allow early advancement after epoch 15
            recommended_level = min(default_level + 1, 5)
            print(f"📈 ADAPTIVE ADVANCE: Level {default_level} → {recommended_level}")
            return recommended_level
    
    return default_level

# Quick commands for level progression
def quick_advance_commands():
    """Quick commands to copy-paste for level advancement"""
    
    print("""
🚀 QUICK ADVANCE COMMANDS:

# 1. Check if ready to advance
ready = check_training_readiness(train_losses[-10:], val_losses[-10:])

# 2. Force advance to next level  
if ready or input("Force advance? (y/n): ").lower() == 'y':
    
    # Override the level function temporarily
    original_get_level = get_loss_level_for_epoch
    
    def get_loss_level_for_epoch_advanced(epoch):
        base_level = original_get_level(epoch) 
        return min(base_level + 1, 5)  # Advance by 1
    
    # Replace the function
    import progressive_loss
    progressive_loss.get_loss_level_for_epoch = get_loss_level_for_epoch_advanced
    
    print("✅ Advanced to next level!")

# 3. Reset to epoch-based scheduling
# progressive_loss.get_loss_level_for_epoch = original_get_level
""")

if __name__ == "__main__":
    print("🎯 Level Progression Guide Loaded!")
    
    quick_advance_commands()
    
    print("""
    
🔄 PROGRESSION WORKFLOW:

1. Train at current level for 5-10 epochs
2. Check: ready = check_training_readiness(train_losses[-10:], val_losses[-10:])  
3. If ready=True, advance level
4. If not ready, continue training
5. Repeat until Level 5

🎮 Or use manual control anytime!
""")