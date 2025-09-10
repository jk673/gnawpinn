#!/usr/bin/env python3
"""
Quick test to verify the physics loss fix works
"""

import torch
from gnawpinn_unified import ComprehensivePhysicsLoss
from torch_geometric.data import Data

# Create test data
device = torch.device('cpu')
batch_size = 50

# 7D node features [x, y, z, normal_x, normal_y, normal_z, area]
x = torch.randn(batch_size, 7, device=device)
# 4D targets [pressure_coeff, tau_x, tau_y, tau_z]
y = torch.randn(batch_size, 4, device=device)
# Dummy edges
edge_index = torch.randint(0, batch_size, (2, batch_size * 2), device=device)
edge_attr = torch.randn(batch_size * 2, 8, device=device)

# Create data object
data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

# Create predictions (same shape as targets)
predictions = torch.randn(batch_size, 4, device=device, requires_grad=True)

# Test physics loss
print("🧪 Testing ComprehensivePhysicsLoss...")
physics_loss = ComprehensivePhysicsLoss().to(device)

try:
    loss_result = physics_loss.compute_loss(predictions, y, data)
    print("✅ Physics loss computation successful!")
    
    print(f"\n📊 Loss components:")
    for key, value in loss_result.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            print(f"  {key}: {value.item():.6f}")
    
    # Test backward pass
    total_loss = loss_result['total_loss']
    total_loss.backward()
    print(f"\n✅ Backward pass successful!")
    print(f"Gradient norm: {torch.norm(predictions.grad):.6f}")
    
    # Verify non-zero penalties
    penalty_keys = [k for k in loss_result.keys() if 'penalty' in k]
    print(f"\n🔥 Penalty terms:")
    for key in penalty_keys:
        if key in loss_result:
            value = loss_result[key]
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                print(f"  {key}: {value.item():.6f} {'✅' if value.item() > 0 else '❌'}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*50)
print("Physics loss test completed!")