#!/bin/bash
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
# echo codepath: $codepath
MAIN_DIR=$codepath # path to the main directory of the code repo
CKPT_DIR="${MAIN_DIR}/checkpoints" # path to the checkpoint directory
# MODEL = "$1" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
# DATASET = "$2"
# MODELS="TIRG ARTEMIS MAAF CLIP4CIR" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL
MODEL="CLIP4CIR" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL

IMG_CORRUPTS="gaussian_noise_filter shot_noise_filter impulse_noise_filter gaussian_blur_filter motion_blur_filter  defocus_blur_filter zoom_blur_filter brightness_filter contrast_filter  pixelate_filter "
IMG_CORRUPT_LEVELS="1 2 3 4 5"

DATASET="cirr" # choose： cirr or fashionIQ 
IMG_CORRUPT="none" # choose： gaussian_noise_filter,motion_blur_filter ,none
IMG_CORRUPT_LEVEL=5 # choose： 1  2  3  4  5
TXT_CORRUPT="none" # choose: character_filter qwerty_filter RemoveChar_filter remove_space_filter misspelling_filter repetition_filter homophones_filter

exp_name="${DATASET}/${MODEL}"

corruption_level="${IMG_CORRUPT}_corruption_level_${IMG_CORRUPT_LEVEL}"
echo "${CKPT_DIR}/${exp_name}/ckpt.pth with ${corruption_level}"


if [ "$DATASET" == "cirr" ] && ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ] || [ "$MODEL" == 'CIRPLANT' ]) ; then
        echo 'with specific args'
        specific_args=(
            --load_image_feature 2048 # loaded feature size (usually: 0)
            --cnn_type 'resnet152'
            --validate "val"
        )
fi
# python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name "corruption_level_0" --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}"

python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name debug --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}" --txt_corrupt ${TXT_CORRUPT}
