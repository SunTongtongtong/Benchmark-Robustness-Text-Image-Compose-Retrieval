#!/bin/bash
EXP=$1
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
# echo codepath: $codepath
MAIN_DIR=$codepath # path to the main directory of the code repo
CKPT_DIR="${MAIN_DIR}/checkpoints" # path to the checkpoint directory
# MODEL = "$1" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
# DATASET = "$2"
MODELS="TEXTONLY" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL /instructBLIP / BLIP4CIR / BLIPv2 / TEXTONLY / RerankFile
# MODELS="TIRG " # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL
# IMG_CORRUPTS="none gaussian_noise_filter shot_noise_filter impulse_noise_filter motion_blur_filter defocus_blur_filter zoom_blur_filter brightness_filter contrast_filter  pixelate_filter jpeg_compression fog snow frost glass_blur elastic_transform" 
# IMG_CORRUPTS="none shot_noise_filter impulse_noise_filter" 
IMG_CORRUPTS="none"
IMG_CORRUPT_LEVELS="5 "
TXT_CORRUPT="none"
for MODEL in ${MODELS}
do
    for IMG_CORRUPT in ${IMG_CORRUPTS}
    do
        for IMG_CORRUPT_LEVEL in ${IMG_CORRUPT_LEVELS}
        do
            DATASET="cirr" # choose： cirr or fashionIQ or circo
            # IMG_CORRUPT="speckle_noise_filter" # choose： gaussian_noise_filter,motion_blur_filter ,none
            # IMG_CORRUPT_LEVEL=1 # choose： 1  2  3  4  5
            exp_name="${DATASET}/${MODEL}_neurips"

            echo "${CKPT_DIR}/${exp_name}/ckpt.pth with  $EXP"            

            if [ "$DATASET" == "cirr" ] && ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ] || [ "$MODEL" == 'CIRPLANT' ]) ; then
                    specific_args=(
                        # --load_image_feature 2048 # loaded feature size (usually: 0)
                        # --cnn_type 'resnet152'
                        --validate "val"
                        --txt_enc_type 'lstm'

                    )
            fi

            python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name "$EXP"  --ckpt  ${CKPT_DIR}/${DATASET}/${MODEL}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}" --txt_corrupt ${TXT_CORRUPT} --studied_split "test"
        done
    done
done
