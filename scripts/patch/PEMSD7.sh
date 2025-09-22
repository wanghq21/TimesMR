export CUDA_VISIBLE_DEVICES=0
model_name=TimesMR

temporal_function='patch'


python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMSD7.npy \
    --model_id PEMSD7_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMSD7.npy \
    --model_id PEMSD7_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMSD7.npy \
    --model_id PEMSD7_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 48 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/PEMS \
    --data_path PEMSD7.npy \
    --model_id PEMSD7_96_96 \
    --model $model_name \
    --data npy \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.5 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 228 \
    --dec_in 228 \
    --c_out 228 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0