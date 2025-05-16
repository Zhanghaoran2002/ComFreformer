export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs1" ]; then
    mkdir -p ./logs1
fi

if [ ! -d "./logs1/ITFnewnew/ETTm2" ]; then
    mkdir -p ./logs1/ITFnewnew/ETTm2
fi

model_name=ComFreformer
seq_len=96
data=ETTm2
attention_type=complex
bw=1
d_deep=48 
for seed in 2025
do
for pred_len in 96
do

python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
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
  --d_ff 128 \
  --attention $attention_type \
  --learning_rate 1e-4 \
  --dropout 0.3 \
  --patience 5 \
  --bw $bw \
  --itr 1  | tee logs1/ITFnewnew/ETTm2/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done