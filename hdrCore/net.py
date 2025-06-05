# uHDR: HDR image editing software
#   Copyright (C) 2021  remi cozot 
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr

# -----------------------------------------------------------------------------
# --- Package hdrCore ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
package hdrCoreconsists of the core classes for HDR imaging.
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# --- class Net ---------------------------------------------------------------
# -----------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self,n_feature, n_output):
        """
        Initialize the neural network with a linear layer followed by batch normalization and a sigmoid activation function.
        Args:
            n_feature (int): Number of input features.
            n_output (int): Number of output features (should be 5 for this network).
        """
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_feature, 5),
            nn.BatchNorm1d(5),
            nn.Sigmoid(),
        )
    # -----------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_feature).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 5).
        """
        return self.layer(x)
    