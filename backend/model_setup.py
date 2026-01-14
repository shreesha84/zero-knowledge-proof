import torch
import torch.nn as nn
import ezkl
import os

# 1. Define a tiny Model (represents your LLM)
class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.layer = nn.Linear(3, 1) # 3 inputs, 1 output (the "secret" weights)

    def forward(self, x):
        return self.layer(x)

def setup_zk_circuit():
    model = TinyModel()
    model.eval()

    # Create dummy input for ONNX export
    x = torch.randn(1, 3)
    
    # Export model to ONNX (the standard for ZKML)
    torch.onnx.export(model, x, "model.onnx", input_names=['input'], output_names=['output'])

    # 2. EZKL Setup
    # Generate settings (defines how to turn math into a circuit)
    ezkl.gen_settings("model.onnx", "settings.json")
    
    # Calibrate (adjusts numerical precision for ZK math)
    # In ZK, we use integers, so we must scale decimal weights carefully
    ezkl.calibrate_settings("input.json", "model.onnx", "settings.json", "resources")

    # Compile the model into a ZK-friendly format
    ezkl.compile_circuit("model.onnx", "compiled.ezkl", "settings.json")

    # 3. Key Generation
    # This creates the 'pk' (Proving Key) for the owner 
    # and 'vk' (Verifying Key) for the public.
    ezkl.get_srs("settings.json")
    ezkl.setup("compiled.ezkl", "vk.key", "pk.key")

if __name__ == "__main__":
    setup_zk_circuit()