#!/bin/bash
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
# echo codepath: $codepath
MAIN_DIR=$codepath # path to the main directory of the code repo
CKPT_DIR="${MAIN_DIR}/checkpoints" # path to the checkpoint directory
# MODEL = "$1" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
# DATASET = "$2"
# MODELS="TIRG ARTEMIS MAAF CLIP4CIR" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL

MODELS="ARTEMIS" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL


# IMG_CORRUPTS="gaussian_noise_filter shot_noise_filter impulse_noise_filter gaussian_blur_filter motion_blur_filter  defocus_blur_filter zoom_blur_filter brightness_filter contrast_filter  pixelate_filter "
IMG_CORRUPTS="jpeg_compression fog snow frost glass_blur elastic_transform"
IMG_CORRUPT_LEVELS="1 2 3 4 5"
for MODEL in ${MODELS}
do
    for IMG_CORRUPT in ${IMG_CORRUPTS}
    do
        for IMG_CORRUPT_LEVEL in ${IMG_CORRUPT_LEVELS}
        do
            DATASET="fashionIQ" # choose： cirr or fashionIQ 
            # IMG_CORRUPT="speckle_noise_filter" # choose： gaussian_noise_filter,motion_blur_filter ,none
            # IMG_CORRUPT_LEVEL=1 # choose： 1  2  3  4  5
            exp_name="${DATASET}/${MODEL}"

            corruption_level="${IMG_CORRUPT}_corruption_level_${IMG_CORRUPT_LEVEL}"
            echo "${CKPT_DIR}/${exp_name}/ckpt.pth"

        # if [ "$MODEL" = "FASHIONVIL" ];then
            specific_args=(
                    # --load_image_feature 2048 # loaded feature size (usually: 0)
                    # --cnn_type 'resnet152'
                    --validate "val"
                    --txt_enc_type 'lstm'
                )
            # if [ "$DATASET" == "cirr" ] ; then
            #         specific_args=(
            #             --load_image_feature 2048 # loaded feature size (usually: 0)
            #             --cnn_type 'resnet152'
            #             --validate "val"
            #         )
            # fi
            # python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name "corruption_level_0" --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}"

            python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name $corruption_level --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}"
        done
    done
done