#read the 7 json files and plot the data with x as the 'num_prompts' and y as the keys in the json file that contain one data.


import json
import matplotlib.pyplot as plt

x = [str(2**i * 10) for i in range(1,8)]
ys = [
 'duration',
 'request_throughput',
 'request_goodput:',
 'output_throughput',
 'mean_ttft_ms',
 'median_ttft_ms',
 'std_ttft_ms',
 'p99_ttft_ms',
 'mean_tpot_ms',
 'median_tpot_ms',
 'std_tpot_ms',
 'p99_tpot_ms',
 'mean_itl_ms',
 'median_itl_ms',
 'std_itl_ms',
 'p99_itl_ms']

y_max=[
    500,
    5.0,
    5.0,
    3000,
    200000,
    200000,
    400000,
    400000,
    200,
    200,
    200,
    500,
    200,
    200,
    1000,
    750
]

for koo in range(20):
    total_output_tokens = []
    data = [[] for _ in range(len(ys))]
    for i in range(1,8):
        json_file = f'num_prompts_{2**i * 10}.json'
        json_full_file = f'benchmark_out/benchmark_{koo}/{json_file}'
        with open(json_full_file, 'r') as file0:
            #read the total_output_tokens
            total_output_tokens.append(json.load(file0)['total_output_tokens'])
        for k, y in enumerate(ys):
            with open(json_full_file, 'r') as file:
                #read only the y data
                data[k].append(json.load(file)[y])
        
    print(data)
    # Create a figure with 14 subplots (7 rows, 2 columns)
    fig, axs = plt.subplots(len(ys)//2, 2, figsize=(12, 24))
    #plot x vs y for each y in ys

    for i, y in enumerate(ys):
        #axs[i//2, i%2].scatter(x, data[i])
        #axs[i//2, i%2].set_title(f'num_prompts vs {y}')
        #axs[i//2, i%2].set_xlabel('num_prompts')
        axs[i//2, i%2].scatter(total_output_tokens, data[i])
        axs[i//2, i%2].set_title(f'total_output_tokens vs {y}')
        axs[i//2, i%2].set_xlabel('total_output_tokens')        


        axs[i//2, i%2].set_ylabel(y)
        #set the y axis max value
        axs[i//2, i%2].set_ylim([0, y_max[i]])
        axs[i//2, i%2].grid()



    #save the plot as a PNG image
    plt.tight_layout()
    plt.savefig(f'benchmark_out/benchmark_{koo}/vertical_plot{koo}_tokens.png')


