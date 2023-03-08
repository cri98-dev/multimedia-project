from libs.stego import *
from libs.evaluation import *
from libs.evaluation import *

main_out_dir = './outputs'
vector_imgs_folder = './inputs/benchmark/vector_imgs'
files_to_hide_folder = './inputs/benchmark/files_to_hide'

embeddings = {'lsb_matching': lsb_matching, 'lsb_replacement': lsb_replacement}


def ask_what_to_do() -> str:
    while True:
        cmd = input("""What do you want to do?

    - hide data into image (h)
    - retrieve data from image (r)
    - plot image channels histogram (p) 
    - save image channels as separate images (s)
    - evaluate PSNR and SSIM between two images (m)
    - benchmark available bits embedding algorithms (b)

cmd letter: """)

        if cmd.lower() == 'h':
            return 'hide'
        elif cmd.lower() == 'r':
            return 'retrieve'
        elif cmd.lower() == 'p':
            return 'plot'
        elif cmd.lower() == 's':
            return 'save'
        elif cmd.lower() == 'm':
            return 'metrics'
        elif cmd.lower() == 'b':
            return 'benchmark'
        print('Invalid option.')

        

def read_embedding_fn() -> str:
    while True:
        out = input('Use LSB-Matching or LSB-Replacement? (m/r): ')
        if out.lower() == 'm':
            return 'lsb_matching'
        elif out.lower() == 'r':
            return 'lsb_replacement'
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


def interactive_iface():
    global main_out_dir
    global vector_imgs_folder
    global files_to_hide_folder

    cmd = ask_what_to_do()

    new_out_dir = read_path(f'Output files main folder path (leave blank for default, i.e. {main_out_dir}): ')
    if new_out_dir != '':
        main_out_dir = new_out_dir

    if cmd == 'hide':
        secret = read_str('Type a secret (Save it. You\'ll need it to retrieve the data): ')
        vector_img_path = read_path('Vector image path: ')
        file_to_hide_path = read_path('File to hide path: ')
        embedding_fn = read_embedding_fn()
        save_map = read_y_n_question_as_bool('Do you want to save the visual map of edited pixels? (y/n): ')
        keep_colors = read_y_n_question_as_bool('Do you want to keep the original colors of the image in the visual map of edited pixels? (y/n): ') if save_map else False
        print_metrics = read_y_n_question_as_bool('Do you want to evaluate diff metrics between original image and output image? (y/n): ')
        print('All needed data gathered. Let\'s go!')
        out_image_path, _ = hide(vector_img_path, file_to_hide_path, secret, embeddings[embedding_fn], main_out_dir, save_map, keep_colors)
        if print_metrics:
            print(f'{evaluate_psnr(vector_img_path, out_image_path) = }')
            print(f'{evaluate_ssim(vector_img_path, out_image_path) = }')
            # print(f'{evaluate_mse(vector_img_path, out_image_path) = }')
    elif cmd == 'retrieve':
        secret = read_str('Type the secret chosen during the hiding phase: ')
        stego_img = read_path('Carrier image path: ')
        print('All needed data gathered. Let\'s go!')
        retrive(stego_img, secret, main_out_dir)
    elif cmd == 'plot':
        img_path = read_path('Image path: ')
        print('All needed data gathered. Let\'s go!')
        save_image_channels_histogram(img_path, main_out_dir)
    elif cmd == 'save':
        img_path = read_path('Image path: ')
        print('All needed data gathered. Let\'s go!')
        save_image_channels(img_path, main_out_dir)
    elif cmd == 'metrics':
        img1_path = read_path('Image1 path: ')
        img2_path = read_path('Image2 path: ')
        print('All needed data gathered. Let\'s go!')
        print(f'{evaluate_psnr(img1_path, img2_path) = }')
        print(f'{evaluate_ssim(img1_path, img2_path) = }')
        # print(f'{evaluate_mse(img1_path, img2_path) = }')
    elif cmd == 'benchmark':
        new_vector_imgs_folder = read_path(f'Path of folder containing the images you\'d like to use as vector images (leave blank for default, i.e. {vector_imgs_folder}): ')
        if new_vector_imgs_folder != '':
            vector_imgs_folder = new_vector_imgs_folder
        new_files_to_hide_folder = read_path(f'Path of folder containing the files you\'d like to hide in the vector images (leave blank for default, i.e. {files_to_hide_folder}): ')
        if new_files_to_hide_folder != '':
            files_to_hide_folder = new_files_to_hide_folder
        secret = read_str('Type a secret: ')
        print('All needed data gathered. Let\'s go!')
        benchmark(secret, embeddings, vector_imgs_folder, files_to_hide_folder, main_out_dir)


if __name__ == '__main__':
    interactive_iface()