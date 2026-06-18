
#!/bin/bash

#!/bin/bash

cd "$(dirname "$0")/../.."

if [ -d .venv ]; then
    source .venv/bin/activate
fi

set +e


# ============================================================
# Uncomment exactly ONE line below, then run:
#     ./run_layer_analysis.sh
#
# Save logs:
#     ./run_layer_analysis.sh 2>&1 | tee out/training_log.txt
# ============================================================

# ============================================================
# IMAGE CLASSIFICATION
# Dataset: cifar-10
# Metric : acc
# ============================================================

python -m ab.nn.train -c img-classification_cifar-10_acc_ResNet -e 50 --layer_analysis

# python -m ab.nn.train -c img-classification_cifar-10_acc_AirNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_AirNext -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_AlexNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_BagNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_ComplexNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_BayesianNet-1 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_ConvNeXt -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_ConvNeXtTransformer -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_DPN68 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_DPN107 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_DPN131 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_DarkNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_DenseNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_Diffuser -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_EfficientNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_FractalNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_GoogLeNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_ICNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_InceptionV3-1 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_MNASNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_MaxVit -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_MobileNetV2 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_MobileNetV3 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_MoE-hetero4-Alex-Dense-Air-Bag -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_RegNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_ShuffleNet -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_SqueezeNet-1 -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_SwinTransformer -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_UNet2D -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_VGG -e 50 --layer_analysis
# python -m ab.nn.train -c img-classification_cifar-10_acc_VisionTransformer -e 50 --layer_analysis

# ============================================================
# IMAGE SEGMENTATION
# Dataset: coco
# Metric : iou
# ============================================================

# python -m ab.nn.train -c img-segmentation_coco_iou_DeepLabV3-1 -e 50 --layer_analysis
# python -m ab.nn.train -c img-segmentation_coco_iou_FCN8s -e 50 --layer_analysis
# python -m ab.nn.train -c img-segmentation_coco_iou_FCN16s -e 50 --layer_analysis
# python -m ab.nn.train -c img-segmentation_coco_iou_FCN32s-1 -e 50 --layer_analysis
# python -m ab.nn.train -c img-segmentation_coco_iou_LRASPP -e 50 --layer_analysis
# python -m ab.nn.train -c img-segmentation_coco_iou_UNet-1 -e 50 --layer_analysis

# ============================================================
# OBJECT DETECTION
# Dataset: coco
# Metric : map
# ============================================================

# python -m ab.nn.train -c obj-detection_coco_map_FasterRCNN -e 50 --layer_analysis
# python -m ab.nn.train -c obj-detection_coco_map_FCOS -e 50 --layer_analysis
# python -m ab.nn.train -c obj-detection_coco_map_RetinaNet -e 50 --layer_analysis
# python -m ab.nn.train -c obj-detection_coco_map_SSDLite -e 50 --layer_analysis

# ============================================================
# TEXT GENERATION
# Dataset: wikitext
# Metric : ppl
# ============================================================

# python -m ab.nn.train -c txt-generation_wikitext_ppl_RNN -e 50 --layer_analysis
# python -m ab.nn.train -c txt-generation_wikitext_ppl_LSTM -e 50 --layer_analysis

# ============================================================
# IMAGE CAPTIONING
# Dataset: coco
# Metric : bleu4
# ============================================================

# python -m ab.nn.train -c img-captioning_coco_bleu4_RESNETLSTM -e 50 --layer_analysis
# python -m ab.nn.train -c img-captioning_coco_bleu4_ResNetTransformer -e 50 --layer_analysis

# ============================================================
# SUPER RESOLUTION
# Dataset: div2k
# Metric : psnr
# ============================================================

# python -m ab.nn.train -c img-super-resolution_div2k_psnr_rlfn -e 50 --layer_analysis



