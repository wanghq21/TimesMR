export CUDA_VISIBLE_DEVICES=0
model_name=TimesMR

temporal_function='patch'


python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch -1 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch -1 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch -1 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS03.npz \
    --model_id PEMS03_96_12 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch -1 \
    --enc_in 358 \
    --dec_in 358 \
    --c_out 358 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0