import torch
import torch.nn as nn

class ElectrostaticGridEncoder(nn.Module):
    
    def __init__(self, grid_dims=(64, 64, 64), hide_dim=256):
        super().__init__()
        
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # D/2, H/2, W/2
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # D/4, H/4, W/4
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  
        )
        
        self.fc = nn.Linear(128, hide_dim)
        
    def forward(self, grid_data):
        """
        Args:
        hide_dim: 3D electrostatic potential grid tensor, shape (batch_size, 1, D, H, W)
        Returns:
        grid_embedding: latent representation of the grid, shape (batch_size, hide_dim)
        """
        features = self.conv3d_layers(grid_data)
        features = features.view(features.size(0), -1)
        grid_embedding = self.fc(features)
        
        return grid_embedding
