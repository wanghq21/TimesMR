export CUDA_VISIBLE_DEVICES=0
model_name=TimesMR
temporal_function='down'


python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch 5 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch 5 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch 5 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_96_96 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --n_patch 5 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 2048 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0