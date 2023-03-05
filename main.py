from iface import *

# N.B.: main_out_dir, vector_imgs_folder, files_to_hide_folder and embeddings are defined in iface.py

do_benchmark = False


if __name__ == '__main__':
    print('\n*******************************************************************************************')
    print('**** This is the non-interactive development interface. End users should run iface.py *****')
    print('*******************************************************************************************', end='\n\n')

    secret = 'abcdef'
    vector_img_path = "./inputs/landscape.jpg"
    file_to_hide_path = "./inputs/ball.png"
    print('-----> data hiding')
    out_image_path, _ = hide(vector_img_path, file_to_hide_path, secret, embeddings['lsb_matching'], main_out_dir, save_map=True, keep_colors=True)
    print('-----> histograms plotting')
    save_image_channels_histogram(vector_img_path, main_out_dir)
    save_image_channels_histogram(out_image_path, main_out_dir)
    # print('-----> channels extraction')
    # save_image_channels(vector_img_path, main_out_dir)
    # save_image_channels(out_image_path, main_out_dir)
    print('-----> data retrieval')
    retrieved_file_saving_path = retrive(out_image_path, secret, main_out_dir)
    if retrieved_file_saving_path:
        print('-----> hashes comparison')
        assert compare_files_hashes(file_to_hide_path, retrieved_file_saving_path) == True, 'Different hashes'
        print('Same hashes')
    print('-----> diff metrics evaluation')
    print(f'{evaluate_psnr(vector_img_path, out_image_path) = }')
    print(f'{evaluate_ssim(vector_img_path, out_image_path) = }')
    # print(f'{evaluate_mse(vector_img_path, out_image_path) = }')
    
    if do_benchmark:
        print('-----> benchmarking')
        benchmark(secret, embeddings, vector_imgs_folder, files_to_hide_folder, main_out_dir)