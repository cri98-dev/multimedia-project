from libs.crypto import *
from libs.bytes import *
from libs.images import *
import os

# N.B.: supported_modes is defined in images.py

terminating_string = '£&$!@#-'



def hide(vector_image_path:str, file_to_hide_path:str, secret:str, embedding_fn, main_out_dir:str='./outputs', save_map=False, keep_colors=True) -> str:
    # embedding phase
    out_dir = os.path.join(main_out_dir, 'stego')
    os.makedirs(out_dir, exist_ok=True)
    vector_image_basename = os.path.basename(vector_image_path).replace(".", "_")
    out_image_path = os.path.join(out_dir, f'{vector_image_basename}_with_hidden_data.png')

    if not image_mode_is_supported(vector_image_path):
        raise Exception(f'The chosen vector image\'s mode in not supported. Try with another image. Supported modes are: {supported_modes}')
    
    key = derive_enc_key(secret)
    enc_bytes = encrypt_file(file_to_hide_path, key, terminating_string)
    bytes_to_bits = convert_bytes_to_bits(enc_bytes)
    editable_channels = get_editable_channels_list(vector_image_path)
    
    print(f'bits to hide:  {len(bytes_to_bits)}')
    print(f'editable bits: {len(editable_channels)}')
    
    if not enough_editable_channels(bytes_to_bits, editable_channels):
        raise Exception(f'File to hide is too large to be hidden in the chosen vector image.')
    
    channels_to_edit = choose_n_channels_to_edit(editable_channels, len(bytes_to_bits), secret)

    # open('edited_channels.txt', 'w+').writelines(list(map(str, channels_to_edit)))

    edited_image_pixels, edited_pixels_map = embedding_fn(vector_image_path, channels_to_edit, bytes_to_bits, keep_colors)

    if save_map:
        edited_pixels_map_saving_path = os.path.join(out_dir, f'{vector_image_basename}_edited_pixels_map.png')
        save_edited_image(vector_image_path, edited_pixels_map_saving_path, edited_pixels_map)

    save_edited_image(vector_image_path, out_image_path, edited_image_pixels)
    return out_image_path, (len(bytes_to_bits), len(editable_channels))



def retrive(edited_image_path:str, secret:str, main_out_dir:str='./outputs') -> str:
    # retrieval phase
    out_dir = os.path.join(main_out_dir, 'retrieved_data')
    os.makedirs(out_dir, exist_ok=True)

    if not image_mode_is_supported(edited_image_path):
        raise Exception(f'The chosen vector image\'s mode in not supported. Try with another image. Supported modes are: {supported_modes}')

    edited_channels_list = get_possibly_edited_channels_list(edited_image_path, secret)

    # open('possibly_edited_channels.txt', 'w+').writelines(list(map(str, edited_channels_list))[:1000])

    hidden_bits = retrieve_hidden_bits_from_edited_image(edited_image_path, edited_channels_list)
    bits_to_bytes = convert_bits_to_bytes(hidden_bits)
    key = derive_enc_key(secret)
    dec_bytes, filename = decrypt_bytes(bits_to_bytes, key, terminating_string)
    if filename is None:
        print('No hidden data found.')
        return None
    retrieved_file_saving_path = os.path.join(out_dir, filename)
    save_bytes_as_file(retrieved_file_saving_path, dec_bytes)
    return retrieved_file_saving_path