
model_path="/home/llm_irs/models_ww39/llama-2-7b-chat/pytorch/dldt/INT8_compressed_weights/"

source ../../llm_internal_test/python-env/bin/activate
# source ../openvino/build/install/setupvars.sh
source ../../ov_uss/build/install/setupvars.sh
prompt_file="../../llm_internal_test/prompts/llama-2-7b-chat_l.jsonl"

# numactl -C 96-143 python ../llm_internal_test/benchmark.py -m ${model_path} -n 3 -t $prompt_file -ic 1024 -s 42 -mc 2 -r result.csv
echo "==================================="
# LD_PRELOAD=${LD_PRELOAD}:/usr/lib/x86_64-linux-gnu/libasan.so.6
# LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/x86_64-linux-gnu/
# numactl -C 48-63 heaptrack python profile_ov_mem.py

# SPR:
numactl -C 96-143 python profile_ov_mem.py

# heaptrack numactl -C 96-143 python profile_ov_mem.py
