from typing import List, Tuple
from PIL import Image
import random
import copy
from matplotlib import pyplot as plt
import os

supported_modes = ['RGB', 'RGBA']



def image_mode_is_supported(img_path:str) ->  bool:
    return Image.open(img_path).mode in supported_modes


def get_editable_channels_list(img_path:str) -> List[Tuple[int, int]]:
    img = Image.open(img_path)
    w, h = img.size
    mode = img.mode
    editable_channels = [] 
    for i in range(w*h):
        for k, _ in enumerate(mode):
            editable_channels.append((i, k))
    # (pixel:int, channel:int)
    # [(0, 0), (0, 1), (0, 2), [(0, 3)], (1, 0), (1, 1), (1, 2), [(1, 3)], ...]
    return editable_channels


def enough_editable_channels(bits_to_hide:str, editable_channels:List[Tuple[int, int]]) -> bool:
    return len(bits_to_hide) <= len(editable_channels)


def choose_n_channels_to_edit(editable_channels:List[Tuple[int, int]], n:int, secret:str) -> List[Tuple[int, int]]:
    random.seed(secret)
    return random.sample(editable_channels, len(editable_channels))[:n]



def lsb_replacement(image_path:str, chosen_channels:str, bits_to_hide:str, keep_colors=True) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    img = Image.open(image_path)
    pixels_list = list(img.getdata())
    edited_pixels_map = copy.deepcopy(pixels_list) if keep_colors else list(Image.new(img.mode, img.size, 'black').getdata())

    for (p,c),b in zip(chosen_channels, bits_to_hide):
        channel_bits = format(pixels_list[p][c], '08b')
        if channel_bits[-1] != b:
            new_channel_bits = channel_bits[:-1] + b
            pixel_tuple_to_list = list(pixels_list[p])
            pixel_tuple_to_list[c] = int(new_channel_bits, 2)
            pixels_list[p] = tuple(pixel_tuple_to_list)
            
            edited_pixels_map[p] = (255, 0, 0, 255) if len(edited_pixels_map[p]) == 4 else (255, 0, 0)

    return pixels_list, edited_pixels_map


def lsb_matching(image_path:str, chosen_channels:str, bits_to_hide:str, keep_colors=True) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    img = Image.open(image_path)
    pixels_list = list(img.getdata())
    edited_pixels_map = copy.deepcopy(pixels_list) if keep_colors else list(Image.new(img.mode, img.size, 'black').getdata())

    for (p,c),b in zip(chosen_channels, bits_to_hide):
        current_channel_value = pixels_list[p][c]
        channel_bits = format(current_channel_value, '08b')
        if channel_bits[-1] != b:
            if current_channel_value == 255 or (random.random() < 0.5 and current_channel_value != 0):
                current_channel_value -= 1
            else:
                current_channel_value += 1
            
            pixel_tuple_to_list = list(pixels_list[p])
            pixel_tuple_to_list[c] = current_channel_value
            pixels_list[p] = tuple(pixel_tuple_to_list)

            edited_pixels_map[p] = (255, 0, 0, 255) if len(edited_pixels_map[p]) == 4 else (255, 0, 0)

    return pixels_list, edited_pixels_map


def save_edited_image(image_path:str, dest_path:str, pixels:List[Tuple[int, ...]]) -> bool:
    img = Image.open(image_path)
    new_img = Image.new(img.mode, img.size)
    new_img.putdata(pixels)
    new_img.save(dest_path)
    print(f'Image saved at {dest_path}')
    return True


def get_possibly_edited_channels_list(edited_img_path:str, secret:str) -> List[Tuple[int, int]]:
    editable_channels = get_editable_channels_list(edited_img_path)
    return choose_n_channels_to_edit(editable_channels, len(editable_channels), secret)



def retrieve_hidden_bits_from_edited_image(edited_img_path:str, edited_channels_list:List[Tuple[int, int]]) -> str:
    pixels_list = list(Image.open(edited_img_path).getdata())
    hidden_bits = ''
    for p,c in edited_channels_list:
        bits = format(pixels_list[p][c], '08b')
        hidden_bits += bits[-1]
    return hidden_bits



def save_image_channels(img_path:str, main_out_dir:str='./outputs') -> bool:
    out_dir = os.path.join(main_out_dir, 'channels')
    os.makedirs(out_dir, exist_ok=True)
    img_basename = os.path.basename(img_path).replace(".", "_")

    img = Image.open(img_path)
    for c in img.mode:
        out_path = os.path.join(out_dir, f'{img_basename}_{c}_channel.png')
        img.getchannel(c).save(out_path)
        print(f'{c} channel saved at {out_path}')
    return True


def save_image_channels_histogram(img_path:str, main_out_dir:str='./outputs') -> bool:
    out_dir = os.path.join(main_out_dir, 'histograms')
    os.makedirs(out_dir, exist_ok=True)
    img_basename = os.path.basename(img_path).replace(".", "_")
    out_path = os.path.join(out_dir, f'{img_basename}_channels_histogram.png')

    img = Image.open(img_path)
    fig = plt.figure(figsize=(18,8))
    for i, c in enumerate(img.mode):
        plt.subplot(2,2,i+1)
        counts = img.getchannel(c).histogram()
        bins = range(len(counts))
        plt.hist(x=bins, bins=bins, weights=counts, color=c.lower() if c != 'A' else 'black')
        plt.title(f'{c}')
        plt.xlabel('pixel value')
        plt.ylabel('count')
    fig.tight_layout(pad=1.5)
    fig.suptitle('Channels Histograms')
    fig.savefig(out_path)
    print(f'histogram saved at {out_path}')
