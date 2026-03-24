#!/bin/bash
#SBATCH -J train
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -o train5.out
#SBATCH -e train5.err
#SBATCH --container-env=PYXIS_LOG_LEVEL=debug,ENROOT_LOG_LEVEL=debug
#   SBATCH --container-image=docker://cr.eu-north1.nebius.cloud/e00vs3fjhrp1wapvyq/josh:6

# tasks per node was 8, then saw https://discuss.pytorch.org/t/error-with-ddp-on-multiple-nodes/195251/3

echo "Slurm job ID $SLURM_JOB_ID"

set -euo pipefail
set -x

# not going to resume training or keep previous runs
rm -rf /checkpoints/*
rm -f /trainedmodel/*


# ===== Shared rendezvous file path in /junk used for coordinating the master node =====
SHAREDFILE="/junk/master_addr_${SLURM_JOB_ID}.txt"

# ===== Rank 0 node writes master hostname =====
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "This node is Rank 0, so writing MASTER_ADDR to $SHAREDFILE, this node's IP is $(hostname -i)"
    hostname -i > "$SHAREDFILE"
fi

# ===== All ranks wait until master writes the file =====
while [ ! -s "$SHAREDFILE" ]; do
    sleep 5
    echo "waiting for shared hostname file to be created"
done

export MASTER_ADDR=$(cat "$SHAREDFILE")
export MASTER_PORT=$(shuf -i 32790-60999 -n 1)  #29500 was already in use!

echo "Rank/procid: $SLURM_PROCID, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

echo "SLURM_NODEID=$SLURM_NODEID"
echo "SLURM_NODENAME=$SLURMD_NODENAME"
echo "HOSTNAME=$(hostname)"


### python port test
if [ "$SLURM_PROCID" -eq 0 ]; then
	echo "this is the master pytorch node"
	python3 -c "import socket; s = socket.socket(); s.bind((\"\", int(\"$MASTER_PORT\"))); print(\"Port $MASTER_PORT is free on this node\"); s.close()"
else
	python3 -c "import socket; s=socket.socket(); s.connect((\"$MASTER_ADDR\", $MASTER_PORT))"
fi


########### TORCHRUN

echo "starting torchrun"

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_TIMEOUT=30

# trying to find weird failure that happens at the end of the training run
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_ASYNC_ERROR_HANDLING=1     # abort all ranks fast
export NCCL_P2P_DISABLE=1 # to diagnose nccl type crash on run before wed 10am 


#max steps was 5000

srun --container-image="docker://cr.eu-north1.nebius.cloud/e00vs3fjhrp1wapvyq/josh:7" \
--container-mounts=/data:/data:ro,/trainedmodel:/trainedmodel:rw,/checkpoints:/checkpoints:rw,/basemodel:/basemodel:ro,/train:/train:ro,/junk:/junk:rw \
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$SLURM_PROCID \
  --rdzv_id=$SLURM_JOBID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  /train/distributed_train.py \
    --model_name_or_path /basemodel \
    --train_file /data/train_alpaca.filtered.jsonl \
    --validation_file /data/test_alpaca.filtered.jsonl \
    --output_dir /checkpoints/imdb-lora \
    --max_steps 10000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --eval_steps 150 \
    --logging_steps 150 \
    --max_seq_length 2048

# seq length of 1024 still results in truncation of some of the entries
# maxsteps was 5000, then modified code to exit early to avoid overfitting and wasting time, so it wont go to 10000
#

echo
echo "pytorch finished, starting lora"
echo

# vllm doesn't need it merged, TODO get rid of this?
# Merge LoRA -> full weights
srun --container-image="docker://cr.eu-north1.nebius.cloud/e00vs3fjhrp1wapvyq/josh:7" \
--container-mounts=/data:/data:ro,/trainedmodel:/trainedmodel:rw,/checkpoints:/checkpoints:rw,/basemodel:/basemodel:ro,/train:/train:ro,/junk:/junk:rw \
python /train/merge_lora.py \
  --base /basemodel \
  --adapter_dir /checkpoints/imdb-lora \
  --out_dir /trainedmodel

echo
echo "Merge complete. Tuned model in /trainedmodel"
echo

# record the files generated
echo
echo "ls of /trainedmodel:"
ls -lh /trainedmodel
echo
echo "ls of /checkpoints/imdb-lora"
ls -lh /checkpoints/imdb-lora
echo
echo

# backup the model
echo "Backing up /trainedmodel to object storage s3://4f4faegf-models/job${SLURM_JOB_ID}"
srun --container-image="docker://cr.eu-north1.nebius.cloud/e00vs3fjhrp1wapvyq/josh:7" \
--container-mounts=/data:/data:ro,/trainedmodel:/trainedmodel:rw,/checkpoints:/checkpoints:rw,/basemodel:/basemodel:ro,/train:/train:ro,/junk:/junk:rw \
aws s3 cp /trainedmodel s3://4f4faegf-models/job${SLURM_JOB_ID}/ --recursive

echo
echo "Finished backup"
echo
echo "Done with sbatch job"
exit

