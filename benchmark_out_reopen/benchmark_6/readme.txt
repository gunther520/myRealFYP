vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-requests --tensor_parallel_size 4
    command = f'python3 benchmarks/benchmark_serving.py ' \
            f'--model meta-llama/Llama-3.1-8B-Instruct ' \
            f'--dataset-name hf ' \
            f'--dataset-path /home/hkngae/test/temp_dataset/output.json ' \
            f'--hf-split train ' \
            f'--num-prompts {prompt_size} ' \
            f'--goodput  ttft:2000  ' \
            f'--save-result ' \
            f'--result-filename {json_file} ' \
            f'--result-dir {result_dir} ' \
            f'--request-rate 80.0 ' \
            f'--burstiness 0.25 ' \
        # f'--max-concurrency 80'
    
