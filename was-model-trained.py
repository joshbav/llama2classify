# examines if the model was actually trained. needed since the trained model performance is horrible

from transformers import AutoModelForCausalLM
import torch

# Load models
trained_model = AutoModelForCausalLM.from_pretrained("/trainedmodel", trust_remote_code=True)
original_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)

# Choose a representative layer to compare
layer_name = "model.layers.0.self_attn.q_proj.weight"

# Get the tensors
trained_tensor = dict(trained_model.named_parameters())[layer_name]
original_tensor = dict(original_model.named_parameters())[layer_name]

# Check if they're the same
are_identical = torch.equal(trained_tensor, original_tensor)
print(f"Trained weights identical to original? {are_identical}")
