## model



基于src/demo.png的测评

- fine-tuned

| model           | reaction | result          | correct |
| --------------- | -------- | --------------- | ------- |
| yolo-small      | 6.4      | bird(0.445)     | OK      |
| yolo-tiny       | 5.9      | bird(0.25)      | Not Bad |
| detr-resnet-50  | 4.6      | broccoli(0.538) | OK      |
| detr-resnet-101 | 7.8      | cat(0.372)      | OK      |
|                 |          |                 |         |
|                 |          |                 |         |



## debug

OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like facebook/detr-resnet-50 is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'