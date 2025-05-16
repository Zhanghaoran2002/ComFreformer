export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/ETTm1" ]; then
    mkdir -p ./logs/ComFreformer/ETTm1
fi

model_name=ComFreformer
seq_len=96
data=ETTm1
attention_type=complex
bw=1
d_deep=96
for seed in  2025
do
for pred_len in 96 192 336 720
do

python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 2 \
  --e_layers 1 \
  --enc_in 7 \
  --des 'Exp' \
  --individual \
  --d_model 128 \
  --d_ff 256 \
  --learning_rate 5e-4 \
  --dropout 0.3 \
  --patience 8 \
  --bw $bw \
  --itr 1  | tee logs/ComFreformer/ETTm1/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done