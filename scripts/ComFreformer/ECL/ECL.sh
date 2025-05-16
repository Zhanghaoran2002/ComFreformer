export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/ECL" ]; then
    mkdir -p ./logs/ComFreformer/ECL
fi

model_name=ComFreformer
seq_len=96
data=electricity
attention_type=complex
bw=1
d_deep=96
for seed in 2025 
do
for pred_len in  96 192 336 720
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 3 \
  --enc_in 321 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 16 \
  --train_epochs 50 \
  --patience 10 \
  --learning_rate 5e-4 \
  --bw $bw \
  --itr 1  > logs/ComFreformer/ECL/$model_name'_'$data'_'$seq_len'_'$pred_len.log 2>&1 &

done
done
