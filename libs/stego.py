from libs.crypto import *
from libs.bytes import *
from libs.images import *
import os
from libs.util import *


terminating_string = '£&$!@#-'



def hide(vector_image_path:str, file_to_hide_path:str, secret:str, embedding_fn, main_out_dir:str='./outputs', save_map=False, keep_colors=True) -> str:
    supported_modes = ['RGB', 'RGBA']
    out_dir = os.path.join(main_out_dir, 'stego')
    os.makedirs(out_dir, exist_ok=True)
    vector_image_basename = os.path.basename(vector_image_path).replace(".", "_")
    out_image_path = os.path.join(out_dir, f'{vector_image_basename}_with_hidden_data.png')

    if not image_mode_is_supported(vector_image_path, supported_modes):
        raise Exception(f'The chosen vector image\'s ({vector_image_path}) mode in not supported. Try with another image. Supported modes are: {supported_modes}')
    
    key = derive_enc_key(secret)
    enc_bytes = encrypt_file(file_to_hide_path, key, terminating_string)
    bytes_to_bits = convert_bytes_to_bits(enc_bytes)
    editable_channels = get_editable_channels_list(vector_image_path)
    
    print(f'bits to hide:  {len(bytes_to_bits)}')
    print(f'editable bits: {len(editable_channels)}')
    
    if not enough_editable_channels(bytes_to_bits, editable_channels):
        raise Exception(f'File {file_to_hide_path} is too big to be hidden in the chosen vector image ({vector_image_path}).')
    
    channels_to_edit = choose_n_channels_to_edit(editable_channels, len(bytes_to_bits), secret)

    edited_image_pixels, edited_pixels_map = embedding_fn(vector_image_path, channels_to_edit, bytes_to_bits, keep_colors)

    if save_map:
        edited_pixels_map_saving_path = os.path.join(out_dir, f'{vector_image_basename}_edited_pixels_map.png')
        save_edited_image(vector_image_path, edited_pixels_map_saving_path, edited_pixels_map)

    save_edited_image(vector_image_path, out_image_path, edited_image_pixels)
    return out_image_path, (len(bytes_to_bits), len(editable_channels))



def retrive(edited_image_path:str, secret:str, main_out_dir:str='./outputs') -> str:
    supported_modes = ['RGB', 'RGBA']
    out_dir = os.path.join(main_out_dir, 'retrieved_data')
    os.makedirs(out_dir, exist_ok=True)

    if not image_mode_is_supported(edited_image_path, supported_modes):
        raise Exception(f'The chosen vector image\'s ({edited_image_path}) mode in not supported. Try with another image. Supported modes are: {supported_modes}')

    edited_channels_list = get_possibly_edited_channels_list(edited_image_path, secret)

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





#############################################################################################################################
########################################################## FOURIER ##########################################################
#############################################################################################################################
def embed_data(vector_image_path:str, img1_to_hide_path:str, img2_to_hide_path:str, secret:str, main_out_dir:str='./outputs', plot_chrominance_diffs:bool=True) -> Tuple[str, List[Tuple[int, int]]]:
    vector_supported_modes = ['RGB']
    hidden_supported_modes = ['L']

    out_dir = os.path.join(main_out_dir, 'stego')
    os.makedirs(out_dir, exist_ok=True)
    vector_image_basename = os.path.basename(vector_image_path).replace(".", "_")
    out_img_path = os.path.join(out_dir, f'{vector_image_basename}_with_hidden_data.png')

    if not image_mode_is_supported(vector_image_path, vector_supported_modes):
        raise Exception(f'The chosen vector image\'s ({vector_image_path}) mode in not supported. Try with another image. Supported modes are: {vector_supported_modes}')

    for img in [img1_to_hide_path, img2_to_hide_path]:
        if img and not image_mode_is_supported(img, hidden_supported_modes):
            raise Exception(f'Image to hide ({img}) mode in not supported. Try with another image. Supported modes are: {hidden_supported_modes}')


    l, a, b = get_lab_repr(vector_image_path)

    fft2_a = np.fft.fft2(a, norm='ortho')
    fft2_b = np.fft.fft2(b, norm='ortho')

    M_a, theta_a = get_magnitude_and_angle(fft2_a)
    M_b, theta_b = get_magnitude_and_angle(fft2_b)

    img_shapes = []

    for img_to_hide_path, magnitude in zip([img1_to_hide_path, img2_to_hide_path], [M_a, M_b]):

        if img_to_hide_path:
            values_to_hide, img_shape = get_pixels_values(img_to_hide_path)

            # 0,7 for png output and norm='ortho';
            # 0,3150 for png output and norm='backward';
            # 0,0.015 for png output and norm='forward';
            values_to_hide = scale_values(values_to_hide, 0, 7)

            len_values_to_hide = len(values_to_hide)
            editable_values = magnitude.size//2

            print('values to hide:', len_values_to_hide)
            print('editable values:', editable_values)

            if len_values_to_hide > editable_values:
                raise Exception(f'File {img_to_hide_path} is too big to be hidden inside the chosen vector image ({vector_image_path}).')


            ####################################### determination of coeffs to replace ###############################################
            one_d_indices = np.argpartition(np.fft.fftfreq(magnitude.size), -len_values_to_hide, axis=None)[::-1][:len_values_to_hide] 
            random.seed(secret)
            random.shuffle(one_d_indices)
            matrix_indices = np.unravel_index(one_d_indices, magnitude.shape)
            rows, cols = matrix_indices
            #########################################################################################################################

            i = 0
            for j,k in zip(rows,cols):
                if i >= len_values_to_hide:
                    break
                magnitude[j][k] = values_to_hide[i]
                magnitude[-j][-k] = values_to_hide[i]
                i+=1
                
            img_shapes.append(img_shape)
        else:
            img_shapes.append(None)


    a_1 = np.fft.ifft2(M_a*np.exp(1j*theta_a), norm='ortho').real
    b_1 = np.fft.ifft2(M_b*np.exp(1j*theta_b), norm='ortho').real


    lab_stego_img = np.dstack([l, a_1, b_1])

    rgb_stego_img = skimage.util.img_as_ubyte(skimage_rgb_conversion(lab_stego_img))
    skimage.io.imsave(out_img_path, rgb_stego_img)
    print(f'File saved at {out_img_path}')

    if plot_chrominance_diffs:
        chrominance_channels_dict={"a": a, 
                                "a' before saving as RGB stego img": a_1, 
                                "a' obtained from RGB stego img": get_lab_repr(out_img_path)[1], 
                                "b": b, 
                                "b' before saving as RGB stego img": b_1, 
                                "b' obtained from RGB stego img": get_lab_repr(out_img_path)[2]}

        save_chrominance_channels_diffs_as_fig(chrominance_channels_dict, suptitle='CIE Lab chrominance channels diffs before and after embedding', main_out_dir=main_out_dir)
        save_chrominance_channels_diffs_as_fig(chrominance_channels_dict, as_magnitudes=True, suptitle='CIE Lab chrominance channels spectral magnitudes diffs before and after embedding', main_out_dir=main_out_dir)
    
    return out_img_path, img_shapes

    #---------------------------------------------------------------------------


def retrieve_data(carrier_image_path:str, hidden_imgs_shapes:List[Tuple[int, int]], secret:str, main_out_dir:str='./outputs') -> List[str]:
    vector_supported_modes = ['RGB']
    out_dir = os.path.join(main_out_dir, 'retrieved_data')
    os.makedirs(out_dir, exist_ok=True)

    if not image_mode_is_supported(carrier_image_path, vector_supported_modes):
        raise Exception(f'The chosen vector image\'s ({carrier_image_path}) mode in not supported. Try with another image. Supported modes are: {vector_supported_modes}')

    _, a_1, b_1 = get_lab_repr(carrier_image_path)

    fft2_a_1 = np.fft.fft2(a_1, norm='ortho')
    fft2_b_1 = np.fft.fft2(b_1, norm='ortho')

    M_a, _ = get_magnitude_and_angle(fft2_a_1)
    M_b, _ = get_magnitude_and_angle(fft2_b_1)

    out_paths = []


    img_count = 0
    for magnitude, shape in zip([M_a, M_b], hidden_imgs_shapes):
        if shape:
            hidden_integers = []
            
            length = np.prod(shape)

            ############################# determination of data-carrying coeffs ##############################
            one_d_indices = np.argpartition(np.fft.fftfreq(magnitude.size), -length, axis=None)[::-1][:length]
            random.seed(secret)
            random.shuffle(one_d_indices)
            matrix_indices = np.unravel_index(one_d_indices, magnitude.shape)
            rows, cols = matrix_indices
            ##################################################################################################

            i = 0
            for j,k in zip(rows,cols):
                if i >= length:
                    break
                hidden_integers.append(magnitude[j][k])
                i+=1

            hidden_integers = scale_values(hidden_integers, 0, 255)

            new_img = Image.new('L', shape, 'black')
            new_img.putdata(hidden_integers)
            retrieved_img_saving_path = os.path.join(out_dir, f'retrieved_from_{chr(ord("a")+img_count)}.png')
            new_img.save(retrieved_img_saving_path)
            out_paths.append(retrieved_img_saving_path)
            print(f'file saved at {retrieved_img_saving_path}')
        else:
            out_paths.append(None)
        img_count += 1

    return out_paths