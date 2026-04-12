import numpy as np
import skimage as ski
# import skvideo as skv
import decord
import cv2
# import av
from abc import ABC, abstractmethod
import ffmpeg

from corrupt.corruption import Corruption
from multiprocessing import Pool

import tqdm

from corrupt.utils import *

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from PIL import Image as PILImage
from io import BytesIO

from skimage.filters import gaussian


#* Image corruption
#* Ref: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py

####################
# Corruption Class
####################

from tqdm.contrib.concurrent import process_map
    
class ImageCorruption(Corruption):
    def __init__(self, device='cpu'):
        self.device = device
        self.c_type = []

    @abstractmethod
    def filter(self, frames, scale, **kwargs):
        raise NotImplementedError

    def run(self, video_path: list, num_workers=1, to_folder=None, **kwargs):
        num_workers = min(num_workers, len(video_path))
        f_list = open(to_folder + '/corrupt_list_{}.txt'.format(self._name), 'w')

        with Pool(num_workers) as p:
            import functools
            _p = functools.partial(self.corrupt, device=self.device, to_folder=to_folder, **kwargs)
            with tqdm.tqdm(total=len(video_path)) as pbar:
                for ret in p.imap(_p, video_path):
                    f_list.writelines([ret])
                    pbar.update()
            # res = p.imap(_p, video_path)
            p.close()
            p.join()

        f_list.close()
        print('corruption done')
                

    def corrupt(self, path: str, scale=None, filter_method=None, width=-1, height=-1, device='cpu', sample=False, sample_method='random', save_video=True, to_folder=None, **kwargs):
        filter_method = self.c_type[np.random.randint(0, len(self.c_type))] if filter_method is None else filter_method
        scale = np.random.randint(1, 6) if scale is None else scale
        
        ddevice = decord.cpu() if device == 'cpu' else decord.gpu()
        vr = decord.VideoReader(path, ddevice, num_threads=4)
        ext = path.split('.')[-1]

        # print('corrupting', path)

        nframe = len(vr)
        duration = vr.get_frame_timestamp(nframe-1)[1] - vr.get_frame_timestamp(0)[0]
        height = vr[0].shape[0] 
        width = vr[0].shape[1]
        channel = vr[0].shape[2]
        pix_fmt = ffmpeg.probe(path)["streams"][0]['pix_fmt']
        codec = ffmpeg.probe(path)["streams"][0]['codec_name']

        # print('nframe', nframe)

        # corrupt frames
        if sample:
            raise NotImplementedError
        else:
            # frames = vr.get_batch(range(nframe)).asnumpy()
            frames = []
            vr.seek(0)
            while True:
                try:
                    frames.append(vr.next().asnumpy())
                except:
                    break
            frames = np.array(frames)

        corrupted_frames = filter_method(frames, scale)

        # print('corrupted_frames', corrupted_frames.shape)

        if save_video:
            if to_folder is None:
                to_path = path.replace('.{}'.format(ext), '_corrupted.{}'.format(ext))
            else:
                to_path = to_folder + '/' + path.split('/')[-1]

            # print('saving to', to_path)

            with av.open(to_path, 'w') as container:
                stream = container.add_stream(codec, round(nframe/duration))
                stream.width = width
                stream.height = height
                stream.pix_fmt = pix_fmt

                for f in corrupted_frames:
                    vf = av.VideoFrame.from_ndarray(f, format='rgb24')
                    for packet in stream.encode(vf):
                        container.mux(packet)

                # Flush stream
                for packet in stream.encode():
                    container.mux(packet)

            # print('saved to', to_path)
        else:
            raise Warning('No use of saving video.')


        return ' '.join((to_path, str(scale), filter_method.__name__)) + '\n'
        # return corrupted_frames

        

#* Noise
def gaussian_noise_filter(frames, scale):
        c = [.08, .12, 0.18, 0.26, 0.38][scale - 1]

        frames = np.array(frames) / 255.
        frames = np.clip(frames + np.random.normal(size=frames.shape, scale=c), 0, 1) * 255
        return frames.astype(np.uint8)
            

def shot_noise_filter(frames, scale, **kwargs):
        c = [60, 25, 12, 5, 3][scale - 1]

        frames = np.array(frames) / 255.
        frames = np.clip(np.random.poisson(frames * c) / float(c), 0, 1) * 255

        return frames.astype(np.uint8)

def impulse_noise_filter(frames, scale, **kwargs):
        c = [.03, .06, .09, .17, .27][scale - 1]

        frames = np.array(frames) / 255.
        frames = np.clip(ski.util.random_noise(frames, mode='s&p', amount=c), 0, 1) * 255

        return frames.astype(np.uint8)


def speckle_noise_filter(frames, scale, **kwargs):
        c = [.15, .2, 0.35, 0.45, 0.6][scale - 1]

        frames = np.array(frames) / 255.
        frames = np.clip(frames + frames * np.random.normal(size=frames.shape, scale=c), 0, 1) * 255

        return frames.astype(np.uint8)


#* Blur
def gaussian_blur_filter(frames, scale, **kwargs):
    from skimage.filters import gaussian
    c = [1, 2, 3, 4, 6][scale - 1]

    frames = gaussian(np.array(frames) / 255., sigma=c, multichannel=True)
    frames = np.clip(frames, 0, 1) * 255

    return frames.astype(np.uint8)

def glass_blur_filter(frames, scale, **kwargs):
    from skimage.filters import gaussian
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][scale - 1]
    hh = frames.shape[1]
    ww = frames.shape[2]

    x = np.uint8(gaussian(np.array(frames) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for n in range(x.shape[0]):
        for i in range(c[2]):
            for h in range(hh - c[1], c[1], -1):
                for w in range(ww - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = min(h + dy, hh), min(w + dx, ww)
                    # swap
                    x[n, h, w], x[n, h_prime, w_prime] = x[n, h_prime, w_prime], x[n, h, w]

    # frames = np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255   # environment hugface
    frames = np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255   # environment searle_2    

    return frames.astype(np.uint8)

def defocus_blur_filter(frames, scale, **kwargs):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][scale - 1]

    frames = np.array(frames) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    c_frames = []

    for x in frames:
        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3*h*w -> h*w*3
        c_frames.append(channels)

    frames = np.clip(np.array(c_frames), 0, 1) * 255

    return frames.astype(np.uint8)


def zoom_blur_filter(frames, scale, **kwargs):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][scale - 1]
    x = (np.array(frames) / 255.).astype(np.float32)
    for xi in x:
        out = np.zeros_like(xi)
        for zoom_factor in c:
            out += clipped_zoom(xi, zoom_factor)

        xi = (xi + out) / (len(c) + 1)    
    frames = np.clip(np.expand_dims(xi,axis=0), 0, 1) * 255

    return frames.astype(np.uint8)


def motion_blur_filter(frames, scale, **kwargs):
    #size - in pixels, size of motion blur
    #angel - in degrees, direction of motion blur
    def apply_motion_blur(image, size, angle):
        k = np.zeros((size, size), dtype=np.float32)
        k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
        k = k * ( 1.0 / np.sum(k) )        
        return cv2.filter2D(image, -1, k) 

    size = [5, 7, 9, 11, 13][scale - 1]
    angle = np.random.uniform(-45, 45)

    frames = np.array([apply_motion_blur(frame, size, angle) for frame in frames])

    return frames.astype(np.uint8)



def _motion_blur_filter(frames, scale, **kwargs):
    from io import BytesIO
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][scale - 1]
    h = frames.shape[1]
    w = frames.shape[2]

    from PIL import Image

    pil_frames = [Image.fromarray(np.uint8(frame), ) for frame in frames]
    outputs = [BytesIO() for _ in pil_frames]
    _ = [f.save(outputs[i], format='PNG') for i,f in enumerate(pil_frames)]

    x = [MotionImage(blob=f.getvalue()) for f in outputs]

    for xi in x:
        xi.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = [cv2.imdecode(np.fromstring(xi.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED) for xi in x]
    x = np.array(x)

    if x.shape != (h, w):
        frames = np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        frames = np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

    return frames.astype(np.uint8)

#* Digital
def brightness_filter(frames, scale, **kwargs):
    c = [.1, .2, .3, .4, .5][scale - 1]

    frames = np.array(frames) / 255.
    frames = [ski.color.rgb2hsv(x) for x in frames]
    for x in frames:
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    frames = [ski.color.hsv2rgb(x) for x in frames]
    frames = np.clip(np.array(frames), 0, 1) * 255

    return frames.astype(np.uint8)

def contrast_filter(frames, scale, **kwargs):
    c = [0.4, .3, .2, .1, .05][scale - 1]

    frames = np.array(frames) / 255.
    means = np.mean(frames, axis=(0, 1), keepdims=True)
    frames = np.clip((frames - means) * c + means, 0, 1) * 255

    return frames.astype(np.uint8)
    

def saturate_filter(frames, scale, **kwargs):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][scale - 1]

    frames = np.array(frames) / 255.
    frames = [ski.color.rgb2hsv(x) for x in frames]
    for x in frames:
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    frames = [ski.color.hsv2rgb(x) for x in frames]
    frames = np.clip(np.array(frames), 0, 1) * 255

    return frames.astype(np.uint8)

def pixelate_filter(frames, scale, **kwargs):
    from PIL import Image as PILImage
    # n, h, w, ch = frames.shape
    h, w, ch = frames[0].shape # shitong
    c = [0.6, 0.5, 0.4, 0.3, 0.25][scale - 1]

    pil_frames = [PILImage.fromarray(np.uint8(frame)) for frame in frames]
    x = [f.resize((int(h * c), int(w * c)), PILImage.BOX) for f in pil_frames]
    x = np.array([np.array(f.resize((w, h), PILImage.BOX)) for f in x]) 
    return x.astype(np.uint8)

    # c = [0.6, 0.5, 0.4, 0.3, 0.25][scale - 1]

    # x = frames[0].resize((int(224 * c), int(224 * c)), PILImage.BOX)
    # x = x.resize((224, 224), PILImage.BOX)
    # return x
    

def jpeg_filter():
    pass


# shitong: from ICLR2019 paper code : https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def jpeg_compression(x, scale=1):
    from PIL import Image as PILImage
    from io import BytesIO
    c = [25, 18, 15, 10, 7][scale - 1]
    output = BytesIO()
    PILImage.fromarray(x[0]).save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    x = np.array(x)
    return [x]


def fog(x, scale=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][scale - 1]
    x = PILImage.fromarray(x[0]).resize((224,224))
    x = np.array(x) / 255.
    h,w,ch = x.shape
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:h, :w][..., np.newaxis]
    
    return [(np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255).astype('uint8')]   


def frost(x, scale=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][scale - 1]
    idx = np.random.randint(5)
    
    filename = ['./corrupt/frost/frost1.jpg', './corrupt/frost/frost2.jpg', './corrupt/frost/frost3.jpg', './corrupt/frost/frost4.jpg', './corrupt/frost/frost5.jpg'][idx]
    frost = cv2.imread(filename)
    # randomly crop and convert to rgb
    x = PILImage.fromarray(x[0]).resize((224,224)) # shitong 

    x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return [(np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)).astype('uint8')]


def snow(x, scale=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][scale - 1]
    # import pdb
    # pdb.set_trace()
    x = PILImage.fromarray(x[0]).resize((224,224)) # shitong 

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='JPEG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    return [(np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255).astype('uint8')]



def glass_blur(x, scale=1):
    # sigma, max_delta, iterations
    x = PILImage.fromarray(x[0]).resize((224,224)) # shitong 

    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][scale - 1]

    # x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    # return [(np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255).astype('uint8')]
    return [(np.clip(gaussian(x / 255., sigma=c[0]), 0, 1) * 255).astype('uint8')]


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, scale=1):
    from scipy.ndimage.interpolation import map_coordinates
    image = PILImage.fromarray(image[0]).resize((224,224)) # shitong 

    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][scale - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return [(np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255).astype('uint8')]






class NoiseCorruption(ImageCorruption):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.c_type = [gaussian_noise_filter, shot_noise_filter, impulse_noise_filter, speckle_noise_filter]
        self._name = 'noise'


class BlurCorruption(ImageCorruption):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.c_type = [gaussian_blur_filter, defocus_blur_filter, zoom_blur_filter, motion_blur_filter]
        self._name = 'blur'

class DigitalCorruption(ImageCorruption):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.c_type = [brightness_filter, contrast_filter, saturate_filter, pixelate_filter]
        self._name = 'digital'