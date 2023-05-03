#!/bin/bash

sh_file_name="script_train.sh"
gpu="0"

config="custom.yml" # if you use other dataset, config/path_config.py should be matched
guid="smiling" # guid should be in utils/text_dic.py


CUDA_VISIBLE_DEVICES=$gpu python main.py  --run_train                    \
                --config $config               \
                --exp ./runs/example           \
                --edit_attr $guid              \
                --do_train 1                   \
                --do_test 1                    \
                --n_train_img 100              \
                --n_test_img 32                \
                --n_iter 5                     \
                --bs_train 1                   \
                --t_0 999                      \
                --n_inv_step 50                \
                --n_train_step 50              \
                --n_test_step 50               \
                --get_h_num 1                  \
                --train_delta_block            \
                --save_x0                      \
                --use_x0_tensor                \
                --lr_training 0.5              \
                --clip_loss_w 1.0              \
                --l1_loss_w 3.0                \
                --add_noise_from_xt            \
                --lpips_addnoise_th 1.2        \
                --lpips_edit_th 0.33           \
                --sh_file_name $sh_file_name   \
                --load_random_noise
                # (optional - if you pass "get LPIPS")
                # --user_defined_t_edit 500      \
                # --user_defined_t_addnoise 200  \

                # (optional - if you pass "precompute")
                # --load_random_noise
