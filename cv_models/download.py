from transformers import DetrImageProcessor, DetrForObjectDetection

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

processor.save_pretrained("F:/models/detr-resnet")
model.save_pretrained("F:/models/detr-resnet")
