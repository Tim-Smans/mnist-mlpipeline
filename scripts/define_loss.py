import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_input', type=str, required=True)
parser.add_argument('--loss_output', type=str, required=True)
parser.add_argument('--optimizer_output', type=str, required=True)
args = parser.parse_args()

# Load model
model = torch.load(args.model_input, weights_only=False)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Save loss and optimizer
torch.save(criterion, args.loss_output)
torch.save(optimizer.state_dict(), args.optimizer_output)