#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE="cpu"
NUM_EPOCHS=10
HEATMAP_BATCH_SIZE=100
GPU_NUMBER=0

DATA_FOLDER=$1
INITIAL_EXAM_LIST_PATH="sample_data/single_exam_before_cropping.pkl"
PATCH_MODEL_PATH="models/sample_patch_model.p"
IMAGE_MODEL_PATH="models/sample_image_model.p"
IMAGEHEATMAPS_MODEL_PATH="models/sample_imageheatmaps_model.p"

CROPPED_IMAGE_PATH="${DATA_FOLDER}/cropped_images"
CROPPED_EXAM_LIST_PATH="${DATA_FOLDER}/cropped_images/cropped_exam_list.pkl"
EXAM_LIST_PATH="${DATA_FOLDER}/data.pkl"
HEATMAPS_PATH="${DATA_FOLDER}/heatmaps"
IMAGE_PREDICTIONS_PATH=$2
IMAGEHEATMAPS_PREDICTIONS_PATH="${DATA_FOLDER}/imageheatmaps_predictions.csv"
export PYTHONPATH=$(pwd):$PYTHONPATH

if [ $ENABLE_GPU ]; then
    DEVICE_TYPE="gpu"
fi

echo "Stage 1: Crop Mammograms"
python3 src/cropping/crop_mammogram.py \
    --input-data-folder $DATA_FOLDER \
    --output-data-folder $CROPPED_IMAGE_PATH \
    --exam-list-path $INITIAL_EXAM_LIST_PATH  \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH  \
    --num-processes $NUM_PROCESSES

echo "Stage 2: Extract Centers"
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
    --data-prefix $CROPPED_IMAGE_PATH \
    --output-exam-list-path $EXAM_LIST_PATH \
    --num-processes $NUM_PROCESSES

echo "Stage 3: Generate Heatmaps"
python3 src/heatmaps/run_producer.py \
    --model-path $PATCH_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --batch-size $HEATMAP_BATCH_SIZE \
    --output-heatmap-path $HEATMAPS_PATH \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER

echo "Stage 4a: Run Classifier (Image)"
python3 src/modeling/run_model.py \
    --model-path $IMAGE_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGE_PREDICTIONS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER

echo "Stage 4b: Run Classifier (Image+Heatmaps)"
python3 src/modeling/run_model.py \
    --model-path $IMAGEHEATMAPS_MODEL_PATH \
    --data-path $EXAM_LIST_PATH \
    --image-path $CROPPED_IMAGE_PATH \
    --output-path $IMAGEHEATMAPS_PREDICTIONS_PATH \
    --use-heatmaps \
    --heatmaps-path $HEATMAPS_PATH \
    --use-augmentation \
    --num-epochs $NUM_EPOCHS \
    --device-type $DEVICE_TYPE \
    --gpu-number $GPU_NUMBER
