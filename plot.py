import json
import os
import matplotlib.pyplot as plt

json_file = "/home/hkngae/test/benchmark_out/benchmark_3/vllm-infqps-Llama-3.1-8B-Instruct-20241125-085228.json"
# Read data from JSON file
with open(json_file, 'r') as file:
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
ax3.set_title('prompt_i vs ttfts')
ax3.set_xlabel('prompt_i')
ax3.set_ylabel('ttfts')

ax4.scatter(num_prompts, itls_avg)
ax4.set_title('prompt_i vs itls_avg')
ax4.set_xlabel('prompt_i')
ax4.set_ylabel('itls_avg')

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG image
output_path = 'benchmark_out/benchmark_3/plot_1280.png'
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))

plt.savefig(output_path)