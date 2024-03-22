# start controller
python3 -m fastchat.serve.controller --host 0.0.0.0 --port 21002

# start web server
srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=fastchat_gradio \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    --pty bash

python3 -m fastchat.serve.gradio_web_server_multi \
        --share \
        --model-list-mode=reload \
        --host 0.0.0.0 \
        --port 21002 \
        --controller-url http://0.0.0.0:21002


# python3 -m fastchat.serve.gradio_web_server \
#     --share \
#     --model-list-mode=reload \
#     --host 0.0.0.0 \
#     --port 21002 \
#     --controller-url http://0.0.0.0:21002

#     --controller-url http://0.0.0.0:21002 \
#     --elo /mnt/petrelfs/linjunyao/projects/FastChat/elo_results/elo_results.pkl \
#     --leaderboard-table-file /mnt/petrelfs/linjunyao/projects/FastChat/elo_results/leaderboard_table.csv \
#     --register /mnt/petrelfs/linjunyao/projects/FastChat/elo_results/register_oai_models.json \
#     --show-terms

# Start model worker
srun \
    --partition=llm_dev2 \
    --quotatype=reserved \
    --job-name=fastchat_internlm7b \
    --gres=gpu:2 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --kill-on-bad-exit=1 \
    python3 -m fastchat.serve.model_worker \
        --controller http://0.0.0.0:21002 \
        --host 0.0.0.0 \
        --port 31001 \
        --worker-address http://0.0.0.0:31001 \
        --num-gpus 2 \
        --model-path internlm/internlm-chat-7b
        # --model-path /mnt/petrelfs/share_data/zhoufengzhe/model_weights/hf_hub/models--internlm--internlm2-chat-7b \
# /mnt/petrelfs/share_data/zhoufengzhe/model_weights/hf_hub/models--internlm--internlm-chat-7b
