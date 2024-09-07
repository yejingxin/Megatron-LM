#!/bin/bash

# Runs the "175B" parameter model
# sudo docker run --ipc=host --shm-size=512m --gpus all -v ~/Megatron-LM:/workspace/Megatron-LM -v ~/wikipedia_data:/workspace/wikipedia_data -v ~/experiments:/workspace/experiments -it nvcr.io/nvidia/pytorch:24.02-py3

#  bash examples/llama/run_llama.sh /workspace/experiments/ckpt /workspace/experiments/tb /workspace/wikipedia_data/gpt2-vocab.json /workspace/wikipedia_data/gpt2-merges.txt /workspace/wikipedia_data/wikipedia/wikipedia-tokenized-for-llama2

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
echo $VOCAB_FILE

MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
echo $MERGE_FILE
DATA_PATH=$5 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

#GPT_MODEL_ARGS=(
#    --num-layers 96 
#    --hidden-size 12288 
#    --num-attention-heads 96 
#    --seq-length 2048 
#    --max-position-embeddings 2048 
#)
GPT_MODEL_ARGS=(
  --seq-length 4096
  --num-layers 32
  --hidden-size 4096
  --ffn-hidden-size 11008
  --num-attention-heads 32
  --swiglu
  --untie-embeddings-and-output-weights 
  --no-position-embedding
  --use-rotary-position-embeddings
  --max-position-embeddings 4096
  --normalization 'RMSNorm'
  --tokenizer-type 'Llama2Tokenizer' 
  --tokenizer-model '/workspace/wikipedia_data/wikipedia/llama-2-7b-megatron-checkpoint/tokenizer.model'
)

#--rampup-batch-size 16 16 5859375 
#--recompute-activations
TRAINING_ARGS=(
  --micro-batch-size 2
  --global-batch-size 32
  --fp16
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-flash-attn
  --ddp-bucket-size 1000000000
  --use-distributed-optimizer
  --no-masked-softmax-fusion
  --attention-softmax-in-fp32
  --overlap-grad-reduce
  --overlap-param-gather
  --train-iters 10
  --log-interval 1
  --eval-iters 0
  --eval-interval 1000
  --weight-decay 0.1 
  --adam-beta1 0.9 
  --adam-beta2 0.95 
  --init-method-std 0.006 
  --clip-grad 1.0 
  --lr 6.0e-5 
  --lr-decay-style cosine 
  --min-lr 6.0e-6
  --lr-warmup-iters 0
  --lr-decay-iters 430000 
)
#--profile

#--num-layers-per-virtual-pipeline-stage 1
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
)

PROFILE_ARGS=(
  --profile
  --use-pytorch-profiler
  --profile-step-start=5
  --profile-step-end=10

)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --num-workers 4
    --split 949,50,1
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
)

EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --log-interval 5
    --save-interval 100000
    --eval-interval 100000
    --save null
    --load null
    --eval-iters null
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

rm -Rf exp
rm -Rf profile/log/*
#nsys profile \
#-s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
#            --force-overwrite=true \
#            -o /workspace/Megatron-LM/profile/llama-rank \
#            --session-new "llama-rank-8gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} --log-interval 1 --tensorboard-dir $TENSORBOARD_LOGS_PATH
    
    #\
    #${EVAL_AND_LOGGING_ARGS[@]}