from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import requests
import numpy as np

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('src/demo.png').convert('RGB')

# image = image.resize((224, 224))
# print(image.size)

# Convert the image to a NumPy array with shape (1, height, width, channels)
# image = np.array(image).reshape(1, 224, 224, 4)

# Normalize the pixel values to the range [0, 1]
# image = image / 255.0

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes

# print results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
