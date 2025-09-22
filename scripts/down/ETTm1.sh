export CUDA_VISIBLE_DEVICES=0
model_name=TimesMR

temporal_function='down'

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.7 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.7 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.7 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --freq t \
    --seq_len 96 \
    --label_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --temporal_function $temporal_function\
    --d_layers 1 \
    --factor 3 \
    --dropout 0.7 \
    --learning_rate 0.001 \
    --n_patch -1 \
    --d_model 512 \
    --d_ff 2048 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1


