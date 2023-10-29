import cv2
from PIL import Image
import torch

cap = cv2.VideoCapture(r"D:\GitHub\Cockroach-video-parse\src\前后颜色标记点.mp4")
ret, frame0 = cap.read()
if not ret:
    exit(0)
size = frame0.shape[:2][::-1]
num = cap.get(7)
fps = int(round(cap.get(5)))

print(frame0.shape)


from transformers import DetrImageProcessor, DetrForObjectDetection
processor = DetrImageProcessor.from_pretrained("F:/models/detr-resnet")
model = DetrForObjectDetection.from_pretrained("F:/models/detr-resnet").eval()

def show_outputs(outpus, frame, size):
    target_sizes = torch.tensor([size])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

cnt = 0
while(ret):
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
    
    print(f"{cnt}:")
    show_outputs(outputs, frame.copy(), size=size)
    