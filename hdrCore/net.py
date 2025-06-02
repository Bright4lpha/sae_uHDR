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
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_feature, 5),
            nn.BatchNorm1d(5),
            nn.Sigmoid(),
        )
    # -----------------------------------------------------------------------------
    def forward(self, x):
        """
        """
        return self.layer(x)
    