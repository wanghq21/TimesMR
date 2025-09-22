export CUDA_VISIBLE_DEVICES=0
model_name=TimesMR
temporal_function='down'


python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.3 \
    --learning_rate 0.001 \
    --n_patch 20 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 192 \
    --e_layers 3 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.3 \
    --learning_rate 0.0001 \
    --n_patch 20 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0



python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 336 \
    --e_layers 3 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.3 \
    --learning_rate 0.0001 \
    --n_patch 20 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar \
    --data_path solar_AL.txt \
    --model_id Solar_96_96 \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 720 \
    --e_layers 3 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.3 \
    --learning_rate 0.0001 \
    --n_patch 20 \
    --d_model 512 \
    --d_ff 512 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 15 \
    --use_norm 0
