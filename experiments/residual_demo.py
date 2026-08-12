import torch
import torch.nn as nn

from model.residual import ResidualConnection


x = torch.randn(2, 5, 8)
sublayer = nn.Linear(8, 8)
residual = ResidualConnection(sublayer)

sublayer_output = sublayer(x)
residual_output = residual(x)

print("Input shape:")
print(x.shape)

print("\nSublayer output shape:")
print(sublayer_output.shape)

print("\nResidual output shape:")
print(residual_output.shape)

print("\nResidual output equals x + sublayer output:")
print(torch.allclose(residual_output, x + sublayer_output))
