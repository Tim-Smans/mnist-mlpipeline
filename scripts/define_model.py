import argparse
import torch
from model import MNISTModel  # Import the shared model definition

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_output', type=str, required=True)
args = parser.parse_args()

# Create and save the model
model = MNISTModel()
torch.save(model, args.model_output)