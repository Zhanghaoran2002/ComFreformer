export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs1" ]; then
    mkdir -p ./logs1
fi

if [ ! -d "./logs1/ComFreformer/ETTh1" ]; then
    mkdir -p ./logs1/ComFreformer/ETTh1
fi

model_name=ComFreformer
seq_len=96
data=ETTh1
attention_type=complex
bw=1
d_deep=48
for seed in 2025
do
for pred_len in 96 192 336 720
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 1 \
  --e_layers 1 \
  --enc_in 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --dropout 0.5 \
  --patience 5 \
  --learning_rate 5e-5 \
  --bw $bw \
  --itr 1  > logs1/ComFreformer/ETTh1/$model_name'_'$data'_'$seq_len'_'$pred_len.log &
done
done