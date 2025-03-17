import onnx 

# Load the ONNX model
model_path = "config/yoeo.onnx"
model = onnx.load(model_path)

# Iterate over all nodes
for node in model.graph.node:
    if node.op_type == "Cast":
        for attr in node.attribute:
            if attr.name == "to" and attr.i == 2:  # 2 is UINT8
                print(f"Fixing Cast node: {node.name}")
                attr.i = 1  # Change to FLOAT32 (1 is FLOAT32 in ONNX)

# Save the modified model
onnx.save(model, "config/yoeo_fixed.onnx")
print("Saved model_fixed.onnx with Cast nodes fixed.")
