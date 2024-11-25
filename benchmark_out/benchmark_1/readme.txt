#vllm serve meta-llama/Llama-3.1-8B-Instruct         --dtype float16         --disable-log-requests --tensor_parallel_size 8 --enable-prefix-caching


for i in range(7,0,-1):
    #command = f'python3 benchmarks/benchmark_serving.py         --model meta-llama/Llama-3.1-8B-Instruct     --dataset-name hf  --dataset-path /home/hkngae/test/temp_dataset/output.json     --hf-split train   --num-prompts {2**i * 10} >> benchmark_out2.txt'
    #command = f'python3 benchmarks/benchmark_serving.py   --model meta-llama/Llama-3.1-8B-Instruct    --dataset-path /home/hkngae/test/temp_dataset/ShareGPT_V3_unfiltered_cleaned_split.json     --hf-split train   --num-prompts {2**i *10} >> benchmark_out4.txt'
    
    command = f'python3 benchmarks/benchmark_serving.py ' \
            f'--model meta-llama/Llama-3.1-8B-Instruct ' \
            f'--dataset-name hf ' \
            f'--dataset-path /home/hkngae/test/temp_dataset/output.json ' \
            f'--hf-split train ' \
            f'--num-prompts {2**i * 10} ' \
            f'--goodput  ttft:2000  ' \
            f'--save-result ' \
            f'--result-dir benchmark_out/benchmark_0_default ' \