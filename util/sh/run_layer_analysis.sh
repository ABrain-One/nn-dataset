#!/bin/bash

cd "$(dirname "$0")/../.."

if [ -d .venv ]; then
    source .venv/bin/activate
fi

set +e

# ============================================================
# Uncomment the desired -c options below, then run:
#     ./run_layer_analysis.sh
#
# Save logs:
#     ./run_layer_analysis.sh 2>&1 | tee out/training_log.txt
# ============================================================

# IMAGE CLASSIFICATION

python -m ab.nn.train -e 50 --layer_analysis \
    -c img-classification_cifar-10_acc_AirNet \
#   -c img-classification_cifar-10_acc_AirNext \
#   -c img-classification_cifar-10_acc_AlexNet \
#   -c img-classification_cifar-10_acc_BagNet \
#   -c img-classification_cifar-10_acc_ComplexNet \
#   -c img-classification_cifar-10_acc_BayesianNet-1 \
#   -c img-classification_cifar-10_acc_ConvNeXt \
#   -c img-classification_cifar-10_acc_ConvNeXtTransformer \
#   -c img-classification_cifar-10_acc_DPN68 \
#   -c img-classification_cifar-10_acc_DPN107 \
#   -c img-classification_cifar-10_acc_DPN131 \
#   -c img-classification_cifar-10_acc_DarkNet \
#   -c img-classification_cifar-10_acc_DenseNet \
#   -c img-classification_cifar-10_acc_Diffuser \
#   -c img-classification_cifar-10_acc_EfficientNet \
#   -c img-classification_cifar-10_acc_FractalNet \
#   -c img-classification_cifar-10_acc_GoogLeNet \
#   -c img-classification_cifar-10_acc_ICNet \
#   -c img-classification_cifar-10_acc_InceptionV3-1 \
#   -c img-classification_cifar-10_acc_MNASNet \
#   -c img-classification_cifar-10_acc_MaxVit \
#   -c img-classification_cifar-10_acc_MobileNetV2 \
#   -c img-classification_cifar-10_acc_MobileNetV3 \
#   -c img-classification_cifar-10_acc_MoE-hetero4-Alex-Dense-Air-Bag \
#   -c img-classification_cifar-10_acc_RegNet \
#   -c img-classification_cifar-10_acc_ResNet \
#   -c img-classification_cifar-10_acc_ShuffleNet \
#   -c img-classification_cifar-10_acc_SqueezeNet-1 \
#   -c img-classification_cifar-10_acc_SwinTransformer \
#   -c img-classification_cifar-10_acc_UNet2D \
#   -c img-classification_cifar-10_acc_VGG \
#   -c img-classification_cifar-10_acc_VisionTransformer \

# IMAGE SEGMENTATION

#   -c img-segmentation_coco_iou_DeepLabV3-1 \
#   -c img-segmentation_coco_iou_FCN8s \
#   -c img-segmentation_coco_iou_FCN16s \
#   -c img-segmentation_coco_iou_FCN32s-1 \
#   -c img-segmentation_coco_iou_LRASPP \
#   -c img-segmentation_coco_iou_UNet-1 \

# OBJECT DETECTION

#   -c obj-detection_coco_map_FasterRCNN \
#   -c obj-detection_coco_map_FCOS \
#   -c obj-detection_coco_map_RetinaNet \
#   -c obj-detection_coco_map_SSDLite \

# TEXT GENERATION

#   -c txt-generation_wikitext_ppl_RNN \
#   -c txt-generation_wikitext_ppl_LSTM \

# IMAGE CAPTIONING

#   -c img-captioning_coco_bleu4_RESNETLSTM \
#   -c img-captioning_coco_bleu4_ResNetTransformer \

# SUPER RESOLUTION

#   -c img-super-resolution_div2k_psnr_RLFN