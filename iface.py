from libs.stego import *
from libs.evaluation import *
from libs.evaluation import *

main_out_dir = './outputs'
vector_imgs_folder = './inputs/benchmark/vector_imgs'
files_to_hide_folder = './inputs/benchmark/files_to_hide'
grayscale_imgs_to_hide_folder = './inputs/benchmark/grayscale_imgs'

lsb_embeddings = {'lsb_matching': lsb_matching, 'lsb_replacement': lsb_replacement}


def ask_what_to_do() -> str:
    while True:
        cmd = input("""What do you want to do?

    - hide data into image (h)
    - retrieve data from image (r)
    - plot image channels histogram (p) 
    - save image channels as separate images (s)
    - evaluate PSNR and SSIM between two images (m)
    - benchmark available embedding algorithms (b)

cmd letter: """)

        if cmd.lower() == 'h':
            return 'hide'
        elif cmd.lower() == 'r':
            return 'retrieve'
        elif cmd.lower() == 'p':
            return 'plot_hists'
        elif cmd.lower() == 's':
            return 'save_channels'
        elif cmd.lower() == 'm':
            return 'metrics'
        elif cmd.lower() == 'b':
            return 'benchmark'
        print('Invalid option.')

        

def read_embedding_fn() -> str:
    while True:
        out = input('Use LSB-Matching, LSB-Replacement, or Fourier? (m/r/f): ')
        if out.lower() == 'm':
            return 'lsb_matching'
        elif out.lower() == 'r':
            return 'lsb_replacement'
        elif out.lower() == 'f':
            return 'fourier'
        print('Invalid option.')


def read_str(prompt:str) -> str:
    out = input(prompt)
    return out


def read_path(prompt:str) -> str:
    path = input(prompt).strip(' \'"')
    return path


def read_y_n_question_as_bool(prompt:str) -> bool:
    while True:
        save = input(prompt)
        if save.lower() == 'y':
            return True
        elif save.lower() == 'n':
            return False
        print('Invalid option.')


def parse_shape_str_as_tuple(input:str):
    out = None
    if input:
        out = tuple(map(int, input.strip().split('x')))
    return out

def interactive_iface():
    global main_out_dir
    global vector_imgs_folder
    global files_to_hide_folder

    cmd = ask_what_to_do()

    new_out_dir = read_path(f'Output files main folder path (leave blank for default, i.e. {main_out_dir}): ')
    if new_out_dir != '':
        main_out_dir = new_out_dir

    if cmd == 'hide':
        embedding_fn = read_embedding_fn()
        secret = read_str('Type a secret (Save it. You\'ll need it to retrieve the data): ')
        vector_img_path = read_path('Vector image path: ')

        if embedding_fn == 'fourier':
            img1_to_hide_path = read_path('Path of grayscale image to hide in chrominance-a channel spectral magnitude (leave blank to hide nothing): ')
            img2_to_hide_path = read_path('Path of grayscale image to hide in chrominance-b channel spectral magnitude (leave blank to hide nothing): ')
            plot_chrominance_diffs = read_y_n_question_as_bool('Do you want to save the figures of the diffs of the chrominance channels before and after the embedding? (y/n): ')
        else:
            file_to_hide_path = read_path('File to hide path: ')
            save_map = read_y_n_question_as_bool('Do you want to save the visual map of edited pixels? (y/n): ')
            keep_colors = read_y_n_question_as_bool('Do you want to keep the original colors of the image in the visual map of edited pixels? (y/n): ') if save_map else False
        
        print_metrics = read_y_n_question_as_bool('Do you want to evaluate diff metrics between original image and output image? (y/n): ')
        
        print('All needed data gathered. Let\'s go!\n')
        
        if embedding_fn == 'fourier':
            out_image_path, hidden_imgs_shapes = embed_data(vector_img_path, img1_to_hide_path, img2_to_hide_path, secret, main_out_dir, plot_chrominance_diffs)
            print(f'Save these shapes: {hidden_imgs_shapes}. You will need them to retrieve the hidden image(s)')
        else:
            out_image_path, _ = hide(vector_img_path, file_to_hide_path, secret, lsb_embeddings[embedding_fn], main_out_dir, save_map, keep_colors)
        
        if print_metrics:
            print(f'{evaluate_ssim(vector_img_path, out_image_path) = }')
            print(f'{evaluate_psnr(vector_img_path, out_image_path) = }')
            # print(f'{evaluate_mse(vector_img_path, out_image_path) = }')
    elif cmd == 'retrieve':
        stego_img = read_path('Carrier image path: ')
        secret = read_str('Type the secret chosen during the hiding phase: ')
        fourier = read_y_n_question_as_bool('Has data to retrieve been hidden via Fourier embedding? (y/n): ')
        if fourier:
            shapes_list = []
            shape_1 = read_str('Shape of image hidden in chrominance-a channel (e.g. 100x100. Leave blank for no image): ')
            shapes_list.append(parse_shape_str_as_tuple(shape_1))
            shape_2 = read_str('Shape of image hidden in chrominance-b channel (e.g. 100x100. Leave blank for no image): ')
            shapes_list.append(parse_shape_str_as_tuple(shape_2))
        print('All needed data gathered. Let\'s go!\n')
        if fourier:
            retrieve_data(stego_img, shapes_list, secret, main_out_dir)
        else:
            retrive(stego_img, secret, main_out_dir)
    elif cmd == 'plot_hists':
        img_path = read_path('Image path: ')
        print('All needed data gathered. Let\'s go!\n')
        save_image_channels_histogram(img_path, main_out_dir)
    elif cmd == 'save_channels':
        img_path = read_path('Image path: ')
        print('All needed data gathered. Let\'s go!\n')
        save_image_channels(img_path, main_out_dir)
    elif cmd == 'metrics':
        img1_path = read_path('Image1 path: ')
        img2_path = read_path('Image2 path: ')
        print('All needed data gathered. Let\'s go!\n')
        print(f'{evaluate_psnr(img1_path, img2_path) = }')
        print(f'{evaluate_ssim(img1_path, img2_path) = }')
        # print(f'{evaluate_mse(img1_path, img2_path) = }')
    elif cmd == 'benchmark':
        new_vector_imgs_folder = read_path(f'Path of folder containing the images you\'d like to use as vector images (leave blank for default, i.e. {vector_imgs_folder}): ')
        
        if new_vector_imgs_folder != '':
            vector_imgs_folder = new_vector_imgs_folder
        
        fourier = read_y_n_question_as_bool('Also benchmark Fourier embedding? (if no, only LSB-based embeddings will be benchmarked) (y/n): ')
        
        prompt = 'Path of folder containing the files you\'d like to hide in the vector images (leave blank for default, i.e. {}): '
        if fourier:
            files_to_hide_folder = grayscale_imgs_to_hide_folder
            prompt = prompt.replace('the files', 'only the grayscale images', 1)
        new_files_to_hide_folder = read_path(prompt.format(files_to_hide_folder))
        
        if new_files_to_hide_folder != '':
            files_to_hide_folder = new_files_to_hide_folder
            
        secret = read_str('Type a secret: ')
        print('All needed data gathered. Let\'s go!\n')
        benchmark(secret, lsb_embeddings, vector_imgs_folder, files_to_hide_folder, main_out_dir, fourier)


if __name__ == '__main__':
    interactive_iface()