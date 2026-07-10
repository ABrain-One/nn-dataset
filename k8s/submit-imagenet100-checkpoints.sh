#!/usr/bin/env bash
# Submit one Kubernetes job per core classifier for ImageNet-100 training.
# Edit NAMESPACE, USER_ID, and HF_TOKEN before running on the cluster.

set -euo pipefail

NAMESPACE="${NAMESPACE:-b-ahsan3919}"
USER_ID="${USER_ID:-1030}"
HF_TOKEN="${HF_TOKEN:-}"
EPOCHS="${EPOCHS:-50}"
DATASET="${DATASET:-imagenet100}"
HOME_DIR="/shared/ssd/home/${NAMESPACE}"

MODELS=(
  AirNet AirNext AlexNet BagNet ComplexNet BayesianNet-1 ConvNeXt ConvNeXtTransformer
  DPN107 DPN131 DPN68 DarkNet DenseNet Diffuser EfficientNet FractalNet GoogLeNet
  ICNet InceptionV3-1 MNASNet MaxVit MoE-hetero4-Alex-Dense-Air-Bag MobileNetV2
  MobileNetV3 RegNet ResNet ShuffleNet SqueezeNet-1 SwinTransformer UNet2D VGG
  VisionTransformer RLFN SwinIR
)

for model in "${MODELS[@]}"; do
  job_name="ckpt-${DATASET}-$(echo "${model}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9-')"
  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
  namespace: ${NAMESPACE}
spec:
  parallelism: 1
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      securityContext:
        runAsUser: ${USER_ID}
        runAsGroup: ${USER_ID}
      containers:
      - name: train
        image: abrainone/ai-linux:cu12.6.3-latest
        imagePullPolicy: Always
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -euo pipefail
          export HOME=${HOME_DIR}
          export HF_HOME=${HOME_DIR}/.cache/huggingface
          export HF_TOKEN='${HF_TOKEN}'
          export PYTHONUNBUFFERED=1
          cd ${HOME_DIR}/nn-dataset
          export PYTHONPATH=${HOME_DIR}/nn-dataset:\$PYTHONPATH
          python3 -m pip install --user -e . -q
          python3 util/py/checkpoints_to_hugging_face.py \
            --dataset ${DATASET} \
            --epoch-train-max ${EPOCHS} \
            --model ${model}
        resources:
          requests:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: 30Gi
          limits:
            nvidia.com/gpu: 1
            cpu: "64"
            memory: 120Gi
        volumeMounts:
        - {mountPath: /dev/shm, name: shm}
        - {mountPath: /shared/ssd, name: shared-ssd}
        - {mountPath: /shared/local/data/${NAMESPACE}, name: local-data}
      restartPolicy: Never
      volumes:
      - name: shm
        emptyDir: {medium: Memory, sizeLimit: 16Gi}
      - name: shared-ssd
        hostPath: {path: /shared/ssd, type: Directory}
      - name: local-data
        hostPath: {path: /shared/local/data/${NAMESPACE}, type: DirectoryOrCreate}
  backoffLimit: 1
EOF
  echo "submitted ${job_name}"
done
