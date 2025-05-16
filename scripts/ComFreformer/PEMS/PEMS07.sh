export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
fi

if [ ! -d "./logs/ComFreformer/PEMS/PEMS07" ]; then
    mkdir -p ./logs/ComFreformer/PEMS/PEMS07
fi

model_name=ComFreformer
seq_len=96
bw=0
d_deep=96
data=PEMS07
seed=2025
attention_type=complex

for pred_len in 12 24 48
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 3 \
  --enc_in 883 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --learning_rate 1e-4 \
  --lradj type3 \
  --batch_size 16 \
  --patience 5 \
  --train_epochs 50 \
  --bw $bw \
  --revin 0 \
  --itr 1  | tee logs/ComFreformer/PEMS/PEMS07/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
for pred_len in 96
do

nohup python -u run.py \
  --seed $seed \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id $data'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --d_deep $d_deep \
  --n_heads 4 \
  --e_layers 3 \
  --enc_in 883 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --learning_rate 1e-4 \
  --lradj type3 \
  --batch_size 16 \
  --patience 5 \
  --train_epochs 50 \
  --bw $bw \
  --revin 0 \
  --itr 1  | tee logs/ComFreformer/PEMS/PEMS07/$model_name'_'$data'_'$seq_len'_'$pred_len.log &

done
