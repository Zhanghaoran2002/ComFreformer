export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/traffic" ]; then
    mkdir -p ./logs/ComFreformer/traffic
fi

model_name=ComFreformer
seq_len=96
data=traffic
attention_type=complex
bw=1
d_deep=96

for seed in 2025
do
for pred_len in 96 192 336 720
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 4 \
  --enc_in 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --attention_type $attention_type \
  --train_epochs 50 \
  --patience 8 \
  --learning_rate 5e-4 \
  --bw $bw \
  --itr 1  > logs/ComFreformer/traffic/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done