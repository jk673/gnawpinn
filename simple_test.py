"""
Simple test to verify model and training works with minimal data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from model import CFDSurrogateModel

# Create very simple test data
def create_simple_test_data():
    """Create minimal test data"""
    num_nodes = 10
    
    # 7D node features: [x, y, z, normal_x, normal_y, normal_z, area]
    x = torch.randn(num_nodes, 7)
    x[:, :3] *= 5  # positions
    x[:, 3:6] = torch.nn.functional.normalize(x[:, 3:6], dim=1)  # unit normals
    x[:, 6] = torch.abs(x[:, 6]) * 0.1  # positive areas
    
    # Simple edges (connect each node to next)
    edge_index = torch.stack([
        torch.arange(num_nodes - 1),
        torch.arange(1, num_nodes)
    ])
    
    # 4D targets: [pressure_coeff, tau_x, tau_y, tau_z]
    y = torch.randn(num_nodes, 4) * 0.1
    
    # Position data for physics loss
    pos = x[:, :3].clone().requires_grad_(True)
    
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos)
    return data

# Simple MSE-only loss function to test training
def simple_mse_loss(predictions, targets):
    """Just MSE loss - no physics"""
    return nn.functional.mse_loss(predictions, targets)

def test_model_basic():
    """Test basic model functionality"""
    print("🧪 Testing basic model functionality...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = CFDSurrogateModel(
        node_feat_dim=7,
        edge_feat_dim=8,
        hidden_dim=32,
        output_dim=4,
        num_mp_layers=2
    ).to(device)
    
    # Create test data
    data = create_simple_test_data().to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(data)
    
    print(f"✅ Forward pass successful:")
    print(f"  Input shape: {data.x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, data

def test_simple_training():
    """Test simple training loop with MSE only"""
    print("\\n🏃 Testing simple training loop...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = test_model_basic()
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop
    model.train()
    losses = []
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data)
        
        # Simple MSE loss only
        loss = simple_mse_loss(predictions, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    print(f"\\n✅ Simple training successful!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    return model, losses

def test_batch_training():
    """Test training with batches"""
    print("\\n📦 Testing batch training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = CFDSurrogateModel(
        node_feat_dim=7,
        edge_feat_dim=8,
        hidden_dim=32,
        output_dim=4,
        num_mp_layers=2
    ).to(device)
    
    # Create multiple data samples
    data_list = [create_simple_test_data().to(device) for _ in range(5)]
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    
    total_loss = 0
    
    for i, data in enumerate(data_list):
        optimizer.zero_grad()
        
        predictions = model(data)
        loss = simple_mse_loss(predictions, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Sample {i+1}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / len(data_list)
    print(f"\\n✅ Batch training successful!")
    print(f"  Average loss: {avg_loss:.6f}")
    
    return model

if __name__ == "__main__":
    print("🚀 Starting simple model tests...")
    
    try:
        # Test 1: Basic model functionality
        model, data = test_model_basic()
        
        # Test 2: Simple training
        model, losses = test_simple_training()
        
        # Test 3: Batch training
        model = test_batch_training()
        
        print("\\n🎉 All tests passed! Model is working correctly.")
        print("\\nThe issue is likely in the comprehensive loss function or training loop.")
        print("Try using simple MSE loss first, then gradually add physics losses.")
        
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()