#!/usr/bin/env python3
"""
Test script for PINN loss functions
"""

try:
    import torch
    import torch.nn.functional as F
    from loss import (
        compute_pressure_gradient, 
        compute_wall_shear_stress,
        pressure_wss_coupling_loss,
        advanced_pinn_loss
    )
    
    def test_pinn_loss_functions():
        """Test the PINN loss functions with synthetic data"""
        print("🔬 Testing PINN loss functions...")
        
        # Create synthetic data
        num_nodes = 50
        spatial_dim = 2
        
        # Positions (2D grid-like)
        pos = torch.randn(num_nodes, spatial_dim) * 2.0
        
        # Create simple edge connectivity
        edges = []
        for i in range(num_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])  # Bidirectional edges
            if i < num_nodes - 3:
                edges.extend([[i, i+3], [i+3, i]])  # Skip connections
        
        edge_index = torch.tensor(edges).T  # [2, E]
        
        # Synthetic predictions [pressure, velocity_x, velocity_y]
        # Create some realistic patterns
        pressure = torch.sin(torch.linspace(0, 3.14, num_nodes)) + 0.1 * torch.randn(num_nodes)
        velocity_x = torch.cos(torch.linspace(0, 3.14, num_nodes)) + 0.1 * torch.randn(num_nodes)
        velocity_y = 0.5 * torch.sin(torch.linspace(0, 6.28, num_nodes)) + 0.1 * torch.randn(num_nodes)
        
        pred = torch.stack([pressure, velocity_x, velocity_y], dim=1)  # [N, 3]
        
        # Create a batch-like object
        class SyntheticBatch:
            def __init__(self, pos, edge_index):
                self.pos = pos
                self.edge_index = edge_index
        
        batch = SyntheticBatch(pos, edge_index)
        
        print(f"📊 Test data: {num_nodes} nodes, {edge_index.shape[1]} edges")
        print(f"    Prediction shape: {pred.shape}")
        print(f"    Position shape: {pos.shape}")
        
        # Test individual functions
        print("\n1️⃣ Testing pressure gradient computation...")
        try:
            pressure_grad_mag, pressure_grad_vec = compute_pressure_gradient(
                pred[:, 0], pos, edge_index
            )
            print(f"    ✅ Pressure gradient magnitude: {pressure_grad_mag.shape} -> {pressure_grad_mag.mean():.4f} ± {pressure_grad_mag.std():.4f}")
            print(f"    ✅ Pressure gradient vector: {pressure_grad_vec.shape}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return False
        
        print("\n2️⃣ Testing WSS computation...")
        try:
            wss_magnitude, wss_vector = compute_wall_shear_stress(
                pred[:, 1:3], pos, edge_index
            )
            print(f"    ✅ WSS magnitude: {wss_magnitude.shape} -> {wss_magnitude.mean():.4f} ± {wss_magnitude.std():.4f}")
            print(f"    ✅ WSS vector: {wss_vector.shape}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return False
        
        print("\n3️⃣ Testing pressure-WSS coupling loss...")
        try:
            coupling_loss, loss_components = pressure_wss_coupling_loss(
                pred, pos, edge_index,
                pressure_weight=1.0, wss_weight=1.0
            )
            print(f"    ✅ Coupling loss: {coupling_loss.item():.6f}")
            print("    📋 Loss components:")
            for key, value in loss_components.items():
                print(f"        {key}: {value:.6f}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return False
        
        print("\n4️⃣ Testing advanced PINN loss...")
        try:
            total_loss, loss_dict = advanced_pinn_loss(
                pred, batch,
                physics_weight=1.0, smoothness_weight=0.1,
                pressure_weight=1.0, wss_weight=1.0
            )
            print(f"    ✅ Total loss: {total_loss.item():.6f}")
            print("    📋 All loss components:")
            for key, value in loss_dict.items():
                print(f"        {key}: {value:.6f}")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return False
        
        print("\n🎉 All PINN loss function tests passed!")
        
        # Test with different scenarios
        print("\n🔄 Testing edge cases...")
        
        # Test with minimal data
        try:
            small_pred = pred[:10, :]
            small_pos = pos[:10, :]
            small_edges = edge_index[:, edge_index[0] < 10]
            small_edges = small_edges[:, small_edges[1] < 10]
            
            if small_edges.shape[1] > 0:
                loss_small, _ = pressure_wss_coupling_loss(small_pred, small_pos, small_edges)
                print(f"    ✅ Small dataset test passed: loss = {loss_small.item():.6f}")
            else:
                print("    ⚠️ Skipped small dataset test (no valid edges)")
        except Exception as e:
            print(f"    ❌ Small dataset test failed: {e}")
        
        return True
    
    if __name__ == "__main__":
        success = test_pinn_loss_functions()
        if success:
            print("\n✅ All tests completed successfully!")
        else:
            print("\n❌ Some tests failed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure PyTorch is installed: pip install torch")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()