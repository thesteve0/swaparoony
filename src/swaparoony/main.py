import gradio as gr
import insightface
from insightface.app import FaceAnalysis


model_path = "models/inswapper_128.onnx"

value = 0
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model(model_path, download=True, download_zip=True)

print("hello world")
