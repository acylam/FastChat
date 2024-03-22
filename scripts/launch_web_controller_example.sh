python3 -m fastchat.serve.controller --host 10.140.60.208 --port 21001

CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-13b-v1.5 --model-name vicuna-13b --controller http://node-01:10002 --host 0.0.0.0 --port 31000 --worker-address http://$(hostname):31000

srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=sh \
    --gres=gpu:4 \
    --ntasks=4 \
    --ntasks-per-node=4 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \

srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=fastchat_controller \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    bash -c "echo \$(hostname)" | xargs -I{} python3 -m fastchat.serve.controller --host {} --port 8000

# Launch controller
srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=fastchat_controller \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    python -m fastchat.serve.controller --host 0.0.0.0 --port 21002

# Launch model workers
srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=fastchat_worker \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    python -m fastchat.serve.register_worker \
        --controller http://0.0.0.0:21001 \
        --worker-name http://HOST-10.140.60.153:31000

        # --model-path /mnt/petrelfs/share_data/zhoufengzhe/model_weights/hf_hub/models--internlm--internlm-chat-7b \

# Launch model workers
# srun \
#     --partition=llm_dev2 \
#     --quotatype=reserved \
#     --job-name=fastchat_worker \
#     --ntasks=1 \
#     --ntasks-per-node=1 \
#     --cpus-per-task=4 \
#     --kill-on-bad-exit=1 \
#     python -m fastchat.serve.vllm_worker \
#         --model-path internlm/internlm2-chat-7b \
#         --model-name internlm-chat-7b \
#         --controller http://node-01:8000 \
#         --host 0.0.0.0 \
#         --port 31000 \
#         --worker-address http://10.140.60.7:31000
        
# Testing service
python -m fastchat.serve.test_message --model internlm-chat-7b --controller http://localhost:31000



# Example

# config.sh
# Get the full path for the directory where this script is located
cd $( dirname -- "$0" ) 
export BLABLADOR_DIR="$(pwd)/FastChat"
export LOGDIR=$BLABLADOR_DIR/logs
export NCCL_P2P_DISABLE=1 # 3090s do not support p2p
export BLABLADOR_CONTROLLER=http://compute-node1.local
export BLABLADOR_CONTROLLER_PORT=21001

# controller.sh
#/bin/bash

cd $BLABLADOR_DIR
source sc_venv_template/activate.sh

# I can leave it open to 0.0.0.0 as this host is not reachable from the internet
python3 fastchat/serve/controller.py --host 0.0.0.0 


# web.sh
#/bin/bash

source config-blablador.sh
cd $BLABLADOR_DIR
source sc_venv_template/activate.sh

python3 fastchat/serve/gradio_web_server.py \
        --share \
        --model-list-mode=reload \
        --host 0.0.0.0 \


# api.sh
#/bin/bash

source config-blablador.sh
cd $BLABLADOR_DIR
source sc_venv_template/activate.sh

python3 fastchat/serve/openai_api_server.py --host 0.0.0.0 --port 8000


# marcoroni-70.slurm
#!/bin/bash
#SBATCH --job-name=Marcoroni-70B
#SBATCH --output=/data/blablador/logs/%j.txt
#SBATCH --error=/data/blablador/logs/%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00 # I like to release the machine every 4 days - gives me time to reevaluate which models I run
#SBATCH --gres=gpu:7

echo "I AM ON "$(hostname) " running Marcoroni-70B on 7 gpus"

export BLABLADOR_DIR="/data/FastChat" # gotta hardcode it here unfortunately
source $BLABLADOR_DIR/config-blablador.sh

cd $BLABLADOR_DIR
source $BLABLADOR_DIR/sc_venv_template/activate.sh

srun python3 $BLABLADOR_DIR/fastchat/serve/model_worker.py \
     --controller $BLABLADOR_CONTROLLER:$BLABLADOR_CONTROLLER_PORT \
     --port 31028 --worker http://$(hostname):31028 \
     --num-gpus 7 \
     --host 0.0.0.0 \
     --model-path /data/FastChat/models/Marcoroni-70B \

