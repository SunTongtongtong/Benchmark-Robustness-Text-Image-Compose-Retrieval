#!/bin/bash
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
# echo codepath: $codepath
MAIN_DIR=$codepath # path to the main directory of the code repo
CKPT_DIR="${MAIN_DIR}/checkpoints" # path to the checkpoint directory
# MODEL = "$1" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
# DATASET = "$2"
# MODELS="TIRG ARTEMIS MAAF CLIP4CIR" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL

MODELS=" ARTEMIS" # choose: ARTEMIS / TIRG / MAAF / CIRPLANT / CLIP4CIR / FASHIONVIL
CAPTION_DIR="${MAIN_DIR}/reasoning_captions/ab_experiments"
# IMG_CORRUPTS="gaussian_noise_filter shot_noise_filter impulse_noise_filter gaussian_blur_filter motion_blur_filter  defocus_blur_filter zoom_blur_filter brightness_filter contrast_filter  pixelate_filter "
# REASON_FILES="remove_reasoning_caption.json numerical_reasoning_caption.json color_reasoning_caption.json size_reasoning_caption.json"
# REASON_FILES="all_numerical_captions.json"
# REASON_FILES="all_numerical_captions.json all_attribute_captions.json all_remove_captions.json all_background_captions.json"
# CAPTION_DIR="${MAIN_DIR}/reasoning_captions/ab_experiments"

REASON_FILES="all_numerical_captions_1_3.json all_numerical_captions_4_10.json"
# REASON_FILES="Finegrained"

IMG_CORRUPTS="none"
IMG_CORRUPT_LEVELS="5"
for MODEL in ${MODELS}
do
    for REASON_FILE in ${REASON_FILES}    
    do
        REASON_FILE_PATH="${CAPTION_DIR}/${REASON_FILE}"
        for IMG_CORRUPT_LEVEL in ${IMG_CORRUPT_LEVELS}
        do
            DATASET="cirr_reason" # choose： cirr or fashionIQ  cirr_reason
            # IMG_CORRUPT="speckle_noise_filter" # choose： gaussian_noise_filter,motion_blur_filter ,none
            # IMG_CORRUPT_LEVEL=1 # choose： 1  2  3  4  5
            exp_name="${DATASET}/${MODEL}"

            corruption_level="${REASON_FILE}_corruption_level_${IMG_CORRUPT_LEVEL}_lstm512_resnet50_counting_in_sub"
            echo "${CKPT_DIR}/${exp_name}/ckpt.pth"

        # if [ "$MODEL" = "FASHIONVIL" ];then
        #     echo "current use fashionvil model"
        #     python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name "debug" --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} 
        #     # config=projects/fashionvil/configs/e2e_composition.yaml \
        #     # model=fashionvil \
        #     # dataset=fashioniq \
        #     # run_type=test \
        #     # checkpoint.resume_file=./save/fashionvil_composition_fashioniq_e2e_pretrain_final/fashionvil_comp_final.pth
            
        # else
        #     echo "Not use fashionvil model"
        #     python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name "debug" --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt ${IMG_CORRUPT} --img_corrupt_level ${IMG_CORRUPT_LEVEL} 
        # fi
          
            # if ([ "$DATASET" == "cirr_reason" ] || [ "$DATASET" == "cirr" ]) && ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ] || [ "$MODEL" == 'CIRPLANT' ]) ; then
            #         specific_args=(
            #             --load_image_feature 2048 # loaded feature size (usually: 0)
            #             --cnn_type 'resnet152'
            #             --validate "val"
            #             --txt_enc_type 'lstm'
            #         )
            # fi

            if ([ "$DATASET" == "cirr_reason" ] || [ "$DATASET" == "cirr" ]) && ([ "$MODEL" == 'TIRG' ] || [ "$MODEL" == 'ARTEMIS' ] || [ "$MODEL" == 'CIRPLANT' ]) ; then
                specific_args=(
                    # --load_image_feature 2048 # loaded feature size (usually: 0)
                    # --cnn_type 'resnet152'
                    --validate "val"
                    --txt_enc_type 'lstm'
                )
            fi

            python main.py --model ${MODEL}  --data_name ${DATASET} --validate "test-val" --exp_name $corruption_level --studied_split "val" --ckpt  ${CKPT_DIR}/${exp_name}/"ckpt.pth" --img_corrupt "none" --img_corrupt_level ${IMG_CORRUPT_LEVEL} "${specific_args[@]}" --reasoning_caption ${REASON_FILE_PATH}
        done
    done
done