# Models

Each model file implements [ModelInterface](../core/interfaces/model_interface.py) to support the setup and inference of a model.

## PaddleOCR

The artificial text filter requires the paddle-ocr environment to be built into the image. Additionally, the angle classifier weights are not on Hugging Face. To ensure there are no runtime errors it is important to download the angle classifier weights in advance. To download it, use:
- **URL:** https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
- **Layout:** Extract the tarball and ensure the model directory contains `inference.pdmodel` and `inference.pdiparams` (the tarball uses different names; copy/rename as needed).
- **Folder:** The two files above should be placed in models/PaddlePaddle/paddle_ocr_cls

If using the standard cosmos_curator_local_workspace for model storage:

```shell
DEST="$HOME/cosmos_curator_local_workspace/models/PaddlePaddle/paddle_ocr_cls"
URL="https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
mkdir -p "$DEST"
curl -sSL -o /tmp/paddleocr_cls.tar "$URL"
tar -xf /tmp/paddleocr_cls.tar -C /tmp
cp /tmp/ch_ppocr_mobile_v2.0_cls_infer/inference.pdmodel "$DEST/inference.pdmodel"
cp /tmp/ch_ppocr_mobile_v2.0_cls_infer/inference.pdiparams "$DEST/inference.pdiparams"
rm -rf /tmp/ch_ppocr_mobile_v2.0_cls_infer /tmp/paddleocr_cls.tar
```
