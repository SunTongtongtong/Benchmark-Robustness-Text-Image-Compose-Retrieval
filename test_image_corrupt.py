# from corrupt.image_corrupt import *
from PIL import Image
import numpy as np

import corrupt.image_corrupt as img_crpt


# import decord

# frames = decord.VideoReader("./corruption_example/sample-5s.mp4", ctx=decord.cpu(0), num_threads=4)
# framimage,es = frames.get_batch(range(len(frames))).asnumpy()
# corrupted_frames = motion_blur_filter(frames, scale=1)

# import pdb
# pdb.set_trace()

# img_path = '/import/sgg-homes/ss014/datasets/NLVR2/images/dev/dev-896-0-img0.png'
# img_path = '/import/sgg-homes/ss014/datasets/NLVR2/images/dev/dev-896-0-img0.png'
# img_path = '/import/sgg-homes/ss014/project/NeurIPS2023_Robustness/train-12987-2-img1.png'
# image = Image.open(img_path).convert('RGB')
# # image.save('dev-896-0-img0.png')
# image = np.array(image)

# # pdb.set_trace()
# # image = zoom_blur_filter([image], scale=5)[0]
# image = impulse_noise_filter([image], scale=5)[0]
# image = Image.fromarray(image)
# # image = Image.fromarray(image)
# image = image.save('zzzzzz.png')

if __name__ == '__main__':
    # IMG_CORRUPTS=["gaussian_noise_filter", "shot_noise_filter", "impulse_noise_filter", "motion_blur_filter", "defocus_blur_filter", "zoom_blur_filter", "brightness_filter", "contrast_filter",  "pixelate_filter", "jpeg_compression", "fog" ,"snow", "frost", "glass_blur", "elastic_transform"]
    IMG_CORRUPTS=["impulse_noise_filter"]
    # for corrupt in IMG_CORRUPTS:
    #     print(corrupt)
    #     img_path = './sample.png'
    #     image = Image.open(img_path).convert('RGB')
    #     image = np.array(image)
    #     img_corrupt_func = getattr(img_crpt,corrupt)
    #     image = img_corrupt_func([image], scale=3)[0]
    #     image = Image.fromarray(image)
    #     image = image.save('./supplementary_imgs/fashion_'+corrupt+'.png')
    # for level in range(1,6):
    level=5
    print(level)
    # print(corrupt)
    corrupt = "impulse_noise_filter"
    img_path = './corrupt/dev-806-2-img1.png'
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)
    img_corrupt_func = getattr(img_crpt,corrupt)
    image = img_corrupt_func([image], scale=level)[0]
    image = Image.fromarray(image)
    # import pdb
    # pdb.set_trace()
    image = image.save('./supplementary_imgs/cirr_Rebuttal_dis4'+corrupt+'_'+str(level)+'.png')