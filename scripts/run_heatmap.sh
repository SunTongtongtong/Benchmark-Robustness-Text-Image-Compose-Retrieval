#!/bin/bash
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
# echo codepath: $codepath
MAIN_DIR=$codepath # path to the main directory of the code repo
CKPT_DIR="${MAIN_DIR}/checkpoints" # path to the checkpoint directory
# MODEL = "$1" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
# DATASET = "$2"
MODELS="TIRG " # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL
# MODELS="TIRG " # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL

# IMG_CORRUPTS="gaussian_noise_filter shot_noise_filter impulse_noise_filter gaussian_blur_filter motion_blur_filter  defocus_blur_filter zoom_blur_filter brightness_filter contrast_filter  pixelate_filter jpeg_compression fog snow frost glass_blur elastic_transform"

# IMG_CORRUPTS="jpeg_compression fog snow frost glass_blur elastic_transform"
IMG_CORRUPTS='none'
IMG_CORRUPT_LEVELS="5"
for MODEL in ${MODELS}
do
    for IMG_CORRUPT in ${IMG_CORRUPTS}
    do
        DATASET="fashionIQ" # choose： cirr or fashionIQ 
        exp_name="${DATASET}/${MODEL}"
        heatmaps () {
        # python ${codepath}/generate_heatmaps.py "${exp_args[@]}" "${specific_args[@]}" \
        python ${codepath}/heatmap_pytorch.py "${exp_args[@]}" "${specific_args[@]}" \
        --studied_split $1 --gradcam \
        --ckpt ${CKPT_DIR}/${exp_name}/$2 \
        --data_name ${DATASET} \
        --model ${MODEL} \
        --img_corrupt ${IMG_CORRUPT} 
        # --img_corrupt_level ${IMG_CORRUPT_LEVEL}
        }
        if ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ]) ; then
            specific_args=(
    # --validate "test-val" # comment this line if you do not have the test split annotations
            --validate "val" # uncomment this line if you do not have the test split annotations
            --txt_enc_type 'lstm' # shitong add for train lstm
            --lstm_hidden_dim 512
            )
        fi

        if [ "$DATASET" == "fashionIQ" ] ; then
            # heatmaps "test" "val/model_best.pth"
            echo "${DATASET}"
            heatmaps "val"  "ckpt.pth"
        elif [ "$DATASET" == "cirr" ] ; then
            heatmaps "val" "ckpt.pth"
            # echo "Cannot generate heatmaps for CIRR (model starts from image features instead of raw images)."
        fi

        # for IMG_CORRUPT_LEVEL in ${IMG_CORRUPT_LEVELS}
        # do
        #     DATASET="cirr" # choose： cirr or fashionIQ 
        #     # IMG_CORRUPT="speckle_noise_filter" # choose： gaussian_noise_filter,motion_blur_filter ,none
        #     # IMG_CORRUPT_LEVEL=1 # choose： 1  2  3  4  5
        #     exp_name="${DATASET}/${MODEL}"

        #     corruption_level="${IMG_CORRUPT}_corruption_level_${IMG_CORRUPT_LEVEL}"
        #     echo "${CKPT_DIR}/${exp_name}/ckpt.pth with  ${corruption_level}"
            

        #     if [ "$DATASET" == "cirr" ] && ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ]) ; then
        #             specific_args=(
        #                 --load_image_feature 2048 # loaded feature size (usually: 0)
        #                 --cnn_type 'resnet152'
        #                 --validate "val"
        #             )
        #     fi

        #     python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name $corruption_level --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}"
        # done
    done
done