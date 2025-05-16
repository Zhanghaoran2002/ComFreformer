export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/Solar" ]; then
    mkdir -p ./logs/ComFreformer/Solar
fi

model_name=ComFreformer
seq_len=96
data=Solar
attention_type=complex
modrelub=0
bw=1
d_deep=48
for seed in 2025 
do
for pred_len in 96 192 336 720
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar_AL.txt \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 2 \
  --e_layers 2 \
  --enc_in 137 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --patience 5 \
  --learning_rate 1e-4 \
  --bw $bw \
  --itr 1  > logs/ComFreformer/Solar/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done