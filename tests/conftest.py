import numpy as np
import onnx
import pytest
from onnx import helper, TensorProto
from PIL import Image


CLASS_NAMES = ["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Rain", "Snow"]


@pytest.fixture
def labels_file(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("\n".join(CLASS_NAMES) + "\n")
    return str(p)


@pytest.fixture
def rgb_image():
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :, 0] = 200  # mostly red
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def onnx_model_file(tmp_path):
    """A minimal ONNX model: input 'input' (1,3,300,300) -> output 'output' (1,6)
    via a fixed Gemm. opset 18. Single-file, no external data."""
    num_classes = len(CLASS_NAMES)
    flat = 3 * 300 * 300
    reshape_shape = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [1, flat]
    )
    weight = helper.make_tensor(
        "W", TensorProto.FLOAT, [flat, num_classes],
        np.zeros(flat * num_classes, dtype=np.float32).tolist(),
    )
    bias_vals = [float(i) for i in range(num_classes)]
    bias = helper.make_tensor("B", TensorProto.FLOAT, [num_classes], bias_vals)
    reshape_node = helper.make_node(
        "Reshape", ["input", "reshape_shape"], ["flat"]
    )
    gemm_node = helper.make_node("Gemm", ["flat", "W", "B"], ["output"])
    graph = helper.make_graph(
        [reshape_node, gemm_node],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 300, 300])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, num_classes])],
        initializer=[reshape_shape, weight, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 18)]
    )
    model.ir_version = 9
    p = tmp_path / "model.onnx"
    onnx.save(model, str(p))
    return str(p)
