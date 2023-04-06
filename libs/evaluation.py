from libs.stego import *
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import numpy as np
from PIL import Image
import pandas as pd
import skimage


def evaluate_ssim(img1_path:str, img2_path:str, axis:int=2):
    return ssim(skimage.io.imread(img1_path), skimage.io.imread(img2_path), channel_axis=axis)

def evaluate_psnr(img1_path:str, img2_path:str):
    return psnr(skimage.io.imread(img1_path), skimage.io.imread(img2_path))

def evaluate_mse(img1_path:str, img2_path:str):
    return mse(skimage.io.imread(img1_path), skimage.io.imread(img2_path))


def benchmark(secret:str, lsb_embeddings:dict, vector_imgs_folder:str='./inputs/benchmark/vector_imgs', files_to_hide_folder:str='./inputs/benchmark/files_to_hide', main_out_dir='./outputs', fourier_embedding=False) -> None:
    vector_img_paths = sorted([os.path.join(vector_imgs_folder, f) for f in os.listdir(vector_imgs_folder)], key=lambda x: np.prod(Image.open(x).size)*len(Image.open(x).mode))
    files_to_hide_paths = sorted([os.path.join(files_to_hide_folder, f) for f in os.listdir(files_to_hide_folder)], key=lambda x: os.stat(x).st_size)
    out_dir = os.path.join(main_out_dir, 'benchmark')
    os.makedirs(out_dir, exist_ok=True)
    out_dict = {'embedding_algorithm':[], 'vector_image':[], 'file_to_hide':[], 'bits_to_hide/editable_bits':[], 'SSIM':[], 'PSNR':[]}
    if fourier_embedding:
        lsb_embeddings['fourier_a'] = None
        lsb_embeddings['fourier_b'] = None
    for file_to_hide_path in files_to_hide_paths:
        if fourier_embedding:
            if not image_mode_is_supported(file_to_hide_path, ['L']):
                print(f'Image to hide ({file_to_hide_path}) is not grayscale. Skipping it.')
                continue
        for vector_img_path in vector_img_paths:
            for embedding_fn_name, embedding_fn in lsb_embeddings.items():
                try:
                    if embedding_fn_name == 'fourier_a':
                        out_image_path, _ = embed_data(vector_img_path, file_to_hide_path, None, secret, main_out_dir, False)
                    elif embedding_fn_name == 'fourier_b':
                        out_image_path, _ = embed_data(vector_img_path, None, file_to_hide_path, secret, main_out_dir, False)
                    else:
                        out_image_path, info = hide(vector_img_path, file_to_hide_path, secret, embedding_fn, main_out_dir)
                    ssim = evaluate_ssim(vector_img_path, out_image_path)
                    psnr = evaluate_psnr(vector_img_path, out_image_path)
                    vector_img = Image.open(vector_img_path)
                    out_dict['embedding_algorithm'].append(f' {embedding_fn_name} ')
                    out_dict['vector_image'].append(f' {os.path.basename(vector_img_path)} ({vector_img.width}x{vector_img.height}, PIL mode: {vector_img.mode}) ')
                    
                    file_to_hide_record = f' {os.path.basename(file_to_hide_path)} ({round(os.stat(file_to_hide_path).st_size / 1024, 2)} KiB) '
                    if fourier_embedding:
                        file_to_hide_record = file_to_hide_record.replace(')', f', {Image.open(file_to_hide_path).width}x{Image.open(file_to_hide_path).height}, PIL mode: {Image.open(file_to_hide_path).mode})')
                    out_dict['file_to_hide'].append(file_to_hide_record)
                    
                    out_dict['bits_to_hide/editable_bits'].append(None if fourier_embedding else f' {info[0]}/{info[1]} ')
                    out_dict['SSIM'].append(f' {ssim} ')
                    out_dict['PSNR'].append(f' {psnr} ')
                except Exception as e:
                    print(e)
        for k in out_dict.keys():
            out_dict[k].append('-')

    out_table = pd.DataFrame.from_dict(out_dict).set_index('embedding_algorithm')
    if fourier_embedding:
        out_table.drop(columns=['bits_to_hide/editable_bits'], inplace=True)
    print(out_table.to_markdown())
    out_path = os.path.join(out_dir, 'benchmark_table.html')
    out_table.to_html(out_path, justify='center', border=2)
    print(f'Table saved at {out_path}')