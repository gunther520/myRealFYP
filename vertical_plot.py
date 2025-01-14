#read the 7 json files and plot the data with x as the 'num_prompts' and y as the keys in the json file that contain one data.


import json
import matplotlib.pyplot as plt

x = [str(2**i * 10) for i in range(1,8)]
ys = [
 'duration',
 'request_throughput',
 'request_goodput:',
 'output_throughput',
 'total_token_throughput',
 'mean_ttft_ms',
 'median_ttft_ms',
 'p99_ttft_ms',
 'mean_tpot_ms',
 'median_tpot_ms',
 'p99_tpot_ms',
 'mean_itl_ms',
 'median_itl_ms',
 'p99_itl_ms']


for koo in range(7):
    data = [[] for _ in range(len(ys))] 
    for i in range(1,8):
        json_file = f'num_prompts_{2**i * 10}.json'
        json_full_file = f'benchmark_out/benchmark_{koo}/{json_file}'
        for k, y in enumerate(ys):
            with open(json_full_file, 'r') as file:
                #read only the y data
                data[k].append(json.load(file)[y])
        
    print(data)
    # Create a figure with 14 subplots (7 rows, 2 columns)
    fig, axs = plt.subplots(7, 2, figsize=(12, 24))
    #plot x vs y for each y in ys

    for i, y in enumerate(ys):
        axs[i//2, i%2].scatter(x, data[i])
        axs[i//2, i%2].set_title(f'num_prompts vs {y}')
        axs[i//2, i%2].set_xlabel('num_prompts')
        axs[i//2, i%2].set_ylabel(y)
        axs[i//2, i%2].grid()


    #save the plot as a PNG image
    plt.tight_layout()
    plt.savefig(f'benchmark_out/benchmark_{koo}/vertical_plot{koo}.png')


