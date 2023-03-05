from libs.stego import *
from typing import List
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import numpy as np
from PIL import Image
import pandas as pd


def evaluate_ssim(img1_path:str, img2_path:str):
    return ssim(np.array(Image.open(img1_path)), np.array(Image.open(img2_path)), channel_axis=2)

def evaluate_psnr(img1_path:str, img2_path:str):
    return psnr(np.array(Image.open(img1_path)), np.array(Image.open(img2_path)))

def evaluate_mse(img1_path:str, img2_path:str):
    return mse(np.array(Image.open(img1_path)), np.array(Image.open(img2_path)))


def benchmark(secret:str, embeddings:dict, vector_imgs_folder:str='./inputs/benchmark/vector_imgs', files_to_hide_folder:str='./inputs/benchmark/files_to_hide', main_out_dir='./outputs') -> None:
    vector_img_paths = sorted([os.path.join(vector_imgs_folder, f) for f in os.listdir(vector_imgs_folder)], key=lambda x: Image.open(x).width*Image.open(x).height*len(Image.open(x).mode))
    files_to_hide_paths = sorted([os.path.join(files_to_hide_folder, f) for f in os.listdir(files_to_hide_folder)], key=lambda x: os.stat(x).st_size)
    out_dir = os.path.join(main_out_dir, 'benchmark')
    os.makedirs(out_dir, exist_ok=True)
    out_dict = {'embedding_algorithm':[], 'vector_image':[], 'file_to_hide':[], 'bits_to_hide/editable_bits':[], 'SSIM':[], 'PSNR':[]}
    for file_to_hide_path in files_to_hide_paths:
        for vector_img_path in vector_img_paths:
            for embedding_fn_name, embedding_fn in embeddings.items():
                try:
                    out_image_path, info = hide(vector_img_path, file_to_hide_path, secret, embedding_fn, main_out_dir)
                    ssim = evaluate_ssim(vector_img_path, out_image_path)
                    psnr = evaluate_psnr(vector_img_path, out_image_path)
                    vector_img = Image.open(vector_img_path)
                    out_dict['embedding_algorithm'].append(f' {embedding_fn_name} ')
                    out_dict['vector_image'].append(f' {os.path.basename(vector_img_path)} ({vector_img.width}x{vector_img.height}, {vector_img.mode}) ')
                    out_dict['file_to_hide'].append(f' {os.path.basename(file_to_hide_path)} ({round(os.stat(file_to_hide_path).st_size / 1024, 2)} KiB) ')
                    out_dict['bits_to_hide/editable_bits'].append(f' {info[0]}/{info[1]} ')
                    out_dict['SSIM'].append(f' {ssim} ')
                    out_dict['PSNR'].append(f' {psnr} ')
                except Exception as e:
                    print(e)
                    break
        for k in out_dict.keys():
            out_dict[k].append('-')

    out_table = pd.DataFrame.from_dict(out_dict).set_index('embedding_algorithm')
    print(out_table.to_markdown())
    out_path = os.path.join(out_dir, 'benchmark_table.html')
    out_table.to_html(out_path, justify='center', border=2)
    print(f'Table saved at {out_path}')