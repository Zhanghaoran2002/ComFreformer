export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/exchange_rate" ]; then
    mkdir -p ./logs/ComFreformer/exchange_rate
fi

model_name=ComFreformer
seq_len=96
data=exchange
bw=1
attention_type=complex
d_deep=96
for seed in  2025
do
for pred_len in 96 192 336 720
do

python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 2 \
  --e_layers 2 \
  --enc_in 8 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --patience 8 \
  --learning_rate 5e-5 \
  --bw $bw \
  --dropout 0.3 \
  --itr 1  | tee logs/ComFreformer/exchange_rate/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done
