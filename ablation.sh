export CUDA_VISIBLE_DEVICES=0
# model_name=Channel_conv
model_name=TimesMR

e_layers=2
d_model=512
d_ff=2048
# npatch=5
for seq in     96 192 336   576 720
do
  for len in    720
  do
    python3 -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/electricity \
      --data_path electricity.csv \
      --model_id ECL_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --freq h \
      --seq_len $seq \
      --label_len 96 \
      --pred_len $len \
      --e_layers $e_layers \
      --d_layers 1 \
      --factor 3 \
      --dropout 0.1 \
      --learning_rate 0.001  \
      --n_patch 5 \
      --d_model $d_model \
      --d_ff $d_ff \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --train_epochs 15 \
      --batch_size 32 \
      --itr 1 \
      --use_norm 1
  done
done




e_layers=2
d_model=512
d_ff=512
# npatch=5
for seq in    96 192 336   576 720
do
  for len in     720
  do
    python3 -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/traffic \
      --data_path traffic.csv \
      --model_id traffic_96_96 \
      --model $model_name \
      --data custom \
      --features M \
      --freq h \
      --seq_len $seq \
      --label_len 96 \
      --pred_len $len \
      --e_layers $e_layers \
      --d_layers 1 \
      --factor 3 \
      --dropout 0.1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate 0.001 \
      --n_patch -1 \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
      --des 'Exp' \
      --lradj type1 \
      --train_epochs 15 \
      --patience 5 \
      --batch_size 32 \
      --itr 1 \
      --use_norm 1
  done
done

