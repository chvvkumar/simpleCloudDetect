import onnxruntime as ort


def test_input_name_and_shape(onnx_model_file):
    sess = ort.InferenceSession(onnx_model_file, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    assert inp.name == "input"
    # Shape may carry symbolic dims; the concrete dims must match (1, 3, 300, 300)
    shape = [d if isinstance(d, int) else None for d in inp.shape]
    assert shape == [1, 3, 300, 300]


def test_output_name(onnx_model_file):
    sess = ort.InferenceSession(onnx_model_file, providers=["CPUExecutionProvider"])
    out = sess.get_outputs()[0]
    assert out.name == "output"
