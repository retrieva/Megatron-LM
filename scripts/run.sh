#!/bin/bash

#$ -cwd
#$ -l node_f=1
#$ -l h_rt=24:00:00
#$ -N retrieva-mlm
#$ -m abe
#$ -o /gs/bs/tge-mc2406/retrieva/logs/stdout.txt
#$ -e /gs/bs/tge-mc2406/retrieva/logs/stderr.txt
#$ -M jiro.nishitoba@retrieva.jp

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FLASH_ATTN=0

export WANDB_ENTITY=retrieva-research
export WANDB_API_KEY=c08d60925405462e3de7c213d6a188810af61457 #jnishi

CHECKPOINT_PATH=/workspace/checkpoints/v3
DATA_PREFIX=/workspace/datasets
DATA_PATH="1 ${DATA_PREFIX}/code_stack_text_document 1 ${DATA_PREFIX}/ja_cc_text_document 1 ${DATA_PREFIX}/ko_wiki_text_document 1 ${DATA_PREFIX}/refinedweb_content_document 1 ${DATA_PREFIX}/zh_wiki_text_document"
TOKENIZER_MODEL=/workspace/tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model
WADB_DIR=./wandb_dir

TRAIN_ITERS=1000000

GPUS_PER_NODE=4
# Change for multinode config

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

WANDB_PROJECT_NAME="bert-128"
#WANDB_PROJECT_NAME="bert-256"
#WANDB_PROJECT_NAME="bert-512"
#WANDB_PROJECT_NAME="bert-1024"
#WANDB_PROJECT_NAME="temp-7-local"


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BERT_ARGS="
    --num-layers 48 \
    --hidden-size 1536 \
    --num-attention-heads 24 \
    --seq-length 128 \
    --max-position-embeddings 2048 \
    --micro-batch-size 128 \
    --global-batch-size 1024 \
    --lr 0.0001 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 450000 \
    --lr-decay-style linear \
    --position-embedding-type rope \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --bf16 \
    --group-query-attention \
    --swiglu \
    --recompute-activations \
    --recompute-granularity selective \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --use-distributed-optimizer \
    --reset-position-ids \
    --transformer-impl transformer_engine \
    --bert-no-binary-head \
    --use-mcore-models \
    --use-flash-attn
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --split 949,50,1 \
    --num-dataset-builder-threads 8
"

OUTPUT_ARGS="
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --wandb-project megatron-bert \
    --wandb-exp-name $WANDB_PROJECT_NAME \
    --wandb-save-dir $WADB_DIR \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --tensorboard-dir ./tensorboard-data/local \
    --log-validation-ppl-to-tensorboard \
    --log-memory-to-tensorboard 
"
cd /gs/bs/tge-mc2406/retrieva/apptainer
apptainer exec -B ../Megatron-LM:/workspace/megatron -B ../llm-jp-tokenizer:/workspace/tokenizer -B ../megatron_processed:/workspace/datasets -B ../checkpoints:/workspace/checkpoints --nv -f -w nvidia-ubuntu/ \
	  torchrun $DISTRIBUTED_ARGS /workspace/megatron/pretrain_bert.py \
	  $BERT_ARGS \
	  $DATA_ARGS \
	  $OUTPUT_ARGS \
	  --distributed-backend nccl \
	  --save $CHECKPOINT_PATH \
	  --load $CHECKPOINT_PATH
