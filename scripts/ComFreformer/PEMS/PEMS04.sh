export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/PEMS/PEMS04" ]; then
    mkdir -p ./logs/ComFreformer/PEMS/PEMS04
fi

model_name=ComFreformer
seq_len=96
bw=0
data=PEMS04
attention_type=complex
d_deep=96
for seed in 2025
do
for pred_len in 12 24 48 96
do

nohup python -u run.py \
  --is_training 1 \
  --seed $seed \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 3 \
  --enc_in 307 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --revin 0  \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --patience 5 \
  --train_epochs 50 \
  --bw $bw \
  --itr 1  | tee logs/ComFreformer/PEMS/PEMS04/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
done
