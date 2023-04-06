from iface import *
from libs.stego import *
from libs.evaluation import *

# N.B.: main_out_dir, vector_imgs_folder, files_to_hide_folder, grayscale_imgs_to_hide_folder and lsb_embeddings are defined in iface.py


lsb = False or True
extra_stuff = False #or True
do_benchmark = False #or True
calculate_metrics = False or True
compare_hashes = False or True



if __name__ == '__main__':
    print('\n*******************************************************************************************')
    print('**** This is the non-interactive development interface. End-users should run iface.py *****')
    print('*******************************************************************************************', end='\n\n')

    if lsb:
        secret = 'abcdef'
        vector_img_path = "./inputs/tucano.png"
        file_to_hide_path = "./inputs/sample.txt"

        print('-----> data hiding')
        out_image_path, _ = hide(vector_img_path, file_to_hide_path, secret, lsb_embeddings['lsb_matching'], main_out_dir, save_map=True, keep_colors=False)
        
        if extra_stuff:
            print('-----> histograms plotting')
            save_image_channels_histogram(vector_img_path, main_out_dir)
            save_image_channels_histogram(out_image_path, main_out_dir)
            print('-----> channels extraction')
            save_image_channels(vector_img_path, main_out_dir)
            save_image_channels(out_image_path, main_out_dir)

        print('-----> data retrieval')
        retrieved_file_saving_path = retrive(out_image_path, secret, main_out_dir)

        if compare_hashes and retrieved_file_saving_path:
            print('-----> hashes comparison')
            assert compare_files_hashes(file_to_hide_path, retrieved_file_saving_path) == True, 'Different hashes'
            print('Same hashes')

        if calculate_metrics:
            print('-----> diff metrics evaluation')
            print(f'{evaluate_ssim(vector_img_path, out_image_path) = }')
            print(f'{evaluate_psnr(vector_img_path, out_image_path) = }')
        
        if do_benchmark:
            print('-----> benchmarking')
            benchmark(secret, lsb_embeddings, vector_imgs_folder, files_to_hide_folder, main_out_dir)
    else:
        vector_image_path = "./inputs/landscape2.jpg"
        img1_to_hide_path = ("./inputs/tucano_G_channel.png", None)[0]
        img2_to_hide_path = ("./inputs/lena.png", None)[0]
        secret = 's3cr3t'

        stego_img_path, hidden_imgs_shapes = embed_data(vector_image_path, img1_to_hide_path, img2_to_hide_path, secret, main_out_dir)

        retrieved_files_paths = retrieve_data(stego_img_path, hidden_imgs_shapes, secret, main_out_dir)

        if calculate_metrics:
            psnr_1 = evaluate_psnr(vector_image_path, stego_img_path)
            ssim_1 = evaluate_ssim(vector_image_path, stego_img_path)
            print('vector_img ---> ssim:', ssim_1, '\tpsnr:', psnr_1)

            if img1_to_hide_path and retrieved_files_paths[0]:
                psnr_2 = evaluate_psnr(img1_to_hide_path, retrieved_files_paths[0])
                ssim_2 = evaluate_ssim(img1_to_hide_path, retrieved_files_paths[0], axis=None)
                print('img1       ---> ssim:', ssim_2, '\tpsnr:', psnr_2)
            if img2_to_hide_path and retrieved_files_paths[1]:
                psnr_3 = evaluate_psnr(img2_to_hide_path, retrieved_files_paths[1])
                ssim_3 = evaluate_ssim(img2_to_hide_path, retrieved_files_paths[1], axis=None)
                print('img2       ---> ssim:', ssim_3, '\tpsnr:', psnr_3)

        if do_benchmark:
            print('-----> benchmarking')
            benchmark(secret, lsb_embeddings, vector_imgs_folder, grayscale_imgs_to_hide_folder, main_out_dir, True)