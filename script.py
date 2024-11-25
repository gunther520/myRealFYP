import subprocess

import json
import os
import matplotlib.pyplot as plt
import time

#vllm serve meta-llama/Llama-3.1-8B-Instruct         --swap-space 16         --disable-log-requests --tensor_parallel_size 8

for i in range(7,0,-1):
    #command = f'python3 benchmarks/benchmark_serving.py         --model meta-llama/Llama-3.1-8B-Instruct     --dataset-name hf  --dataset-path /home/hkngae/test/temp_dataset/output.json     --hf-split train   --num-prompts {2**i * 10} >> benchmark_out2.txt'
    #command = f'python3 benchmarks/benchmark_serving.py   --model meta-llama/Llama-3.1-8B-Instruct    --dataset-path /home/hkngae/test/temp_dataset/ShareGPT_V3_unfiltered_cleaned_split.json     --hf-split train   --num-prompts {2**i *10} >> benchmark_out4.txt'
    
    result_dir = f'benchmark_out/benchmark_4'
    json_file = f"num_prompts_{2**i * 10}.json"


    command = f'python3 benchmarks/benchmark_serving.py ' \
            f'--model meta-llama/Llama-3.1-8B-Instruct ' \
            f'--dataset-name hf ' \
            f'--dataset-path /home/hkngae/test/temp_dataset/output.json ' \
            f'--hf-split train ' \
            f'--num-prompts {2**i * 10} ' \
            f'--goodput  ttft:2000  ' \
            f'--save-result ' \
            f'--result-filename {json_file} ' \
            f'--result-dir {result_dir} ' \
        #f'--num-prompts {2**i * 10} ' \
            
    
    subprocess.run(command, shell=True)

    json_full_file = result_dir + '/' + json_file
    # Read data from JSON file

    #wait until the file exists in the directory before reading it
    while not os.path.exists(json_full_file):
        time.sleep(1)  # Wait for 1 second before checking again

    with open(json_full_file, 'r') as file:
        data = json.load(file)


    # Assuming the JSON data is in the format: {"x": [...], "y": [...]}


    input_lens = data['input_lens']
    num_prompts = [i+1 for i in range(len(input_lens))]
    ttfts = data['ttfts']
    itls = data['itls']

    # itls is a list of lists, we get the average of each list and print it
    itls_avg = [sum(i)/len(i) if len(i) > 0 else 0 for i in itls]

    # Create a figure with 4 subplots (2 rows, 2 columns)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    ax1.scatter(input_lens, ttfts)
    ax1.set_title('input_lens vs ttfts')
    ax1.set_xlabel('input_lens')
    ax1.set_ylabel('ttfts')

    ax2.scatter(input_lens, itls_avg)
    ax2.set_title('input_lens vs itls_avg')
    ax2.set_xlabel('input_lens')
    ax2.set_ylabel('output_lens')


    ax3.scatter(num_prompts, ttfts)
    ax3.set_title(f'prompt_i (Total:{2**i * 10}) vs ttfts')
    ax3.set_xlabel('prompt_i')
    ax3.set_ylabel('ttfts')

    ax4.scatter(num_prompts, itls_avg)
    ax4.set_title(f'prompt_i (Total:{2**i * 10})vs itls_avg')
    ax4.set_xlabel('prompt_i')
    ax4.set_ylabel('itls_avg')

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG image
    output_path = result_dir + f'/plot_{2**i * 10}.png'
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    plt.savefig(output_path)