export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/weather" ]; then
    mkdir -p ./logs/ComFreformer/weather
fi

model_name=ComFreformer
seq_len=96
bw=0
d_deep=96
data=weather
seed=2025
attention_type=complex
for pred_len in 96 192 336 720
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 3 \
  --enc_in 21 \
  --des 'Exp' \
  --d_model 64 \
  --d_ff 64 \
  --individual \
  --dropout 0.3  \
  --attention_type $attention_type \
  --learning_rate 3e-4 \
  --lradj type3 \
  --patience 6 \
  --bw $bw \
  --itr 1  | tee logs/ComFreformer/weather/$model_name'_'$data'_'$seq_len'_'$pred_len.log &
done
