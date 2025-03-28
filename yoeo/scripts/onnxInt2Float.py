import onnx
import argparse

def fix_cast(model_path, output_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Iterate over all nodes
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 2:  # 2 is UINT8
                    print(f"Fixing Cast node: {node.name}")
                    attr.i = 1  # Change to FLOAT32 (1 is FLOAT32 in ONNX)

    # Save the modified model
    onnx.save(model, output_path)
    print(f"Saved model with fixed Cast nodes to {output_path}.")

def fix_cast2(model_path, output_path = "model_fixed.onnx"):
    model = onnx.load(model_path)

    # Iterate over all nodes to change Cast operations to float32
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 2:  # 2 is UINT8
                    print(f"Fixing Cast node: {node.name}")
                    attr.i = 1  # Change to FLOAT32 (1 is FLOAT32 in ONNX)

    # Change the output tensor named "Segmentations" to float32
    for output in model.graph.output:
        if output.name == "Segmentations":
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            print(f"Changed output '{output.name}' to float32.")

    # Save the modified model
    onnx.save(model, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix Cast nodes in an ONNX model.")
    parser.add_argument("model_path", type=str, help="Path to the input ONNX model.")
    parser.add_argument("output_path", type=str, help="Path to save the fixed ONNX model.")

    args = parser.parse_args()

    # Call the function to fix the Cast nodes
    fix_cast(args.model_path, args.output_path)
