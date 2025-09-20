from hailort import InferenceRunner
runner = InferenceRunner("yolov8n.hef")
input_tensor = preprocess("image.jpg")
output = runner.infer(input_tensor)
results = postprocess(output)
print(results)
