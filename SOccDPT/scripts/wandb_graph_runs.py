import wandb
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

api = wandb.Api()

runs_list = [

    ("pw22-sbn-01", "SOccDPT_V3_dpt_swin2_tiny_256_idd", "zyigujaa", "SOccDPT_{V1}"),
    ("pw22-sbn-01", "SOccDPT_V3_dpt_swin2_tiny_256_idd", "cq3j88p0", "SOccDPT_{V1}"),

    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "q0kbdqw5", "SOccDPT_{V2}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "z6q2qcmr", "SOccDPT_{V2}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "1ylk1phx", "SOccDPT_{V2}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "qay53yov", "SOccDPT_{V2}"),

    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "9wltqnkb", "SOccDPT_{V2}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "to209gpm", "SOccDPT_{V2}"),


    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "yrd9xkse", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "oxdknlzn", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "tmgbbmtl", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "ll84baru", "SOccDPT_{V3}"),

]
params_list = [
    '_step',
    'epoch',
    'learning rate',
    'abs_rel',
    'sq_rel',
    'rmse',
    'rmse_log',
    'a1',
    'a2',
    'a3',
    'iou',
]

data = {}

# Check if data.pkl exists
if os.path.exists("data.pkl"):
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
        print("Loaded data.pkl")
        print(data)
else:
    for p in params_list:
        data[p] = {}

print(data.keys())

for (entity, project, run_id, display_name) in tqdm(runs_list):

    if display_name not in data['epoch']:
        run = api.run(f"{entity}/{project}/{run_id}")
        for p in params_list:
            data[p][display_name] = run.history(keys=[p, ])

        # Save data.pkl
        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)
            print("Saved data.pkl")

# Plot one graph per parameter
# Use seperate images for each parameter

runs_list = [

    # ("pw22-sbn-01", "SOccDPT_V3_dpt_swin2_tiny_256_idd", "zyigujaa", "SOccDPT_{V1}"),
    # ("pw22-sbn-01", "SOccDPT_V3_dpt_swin2_tiny_256_idd", "cq3j88p0", "SOccDPT_{V1}"),

    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "q0kbdqw5", "SOccDPT_{V2}"),
    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "z6q2qcmr", "SOccDPT_{V2}"),
    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "1ylk1phx", "SOccDPT_{V2}"),
    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "qay53yov", "SOccDPT_{V2}"),

    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "9wltqnkb", "SOccDPT_{V2}"),
    # ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "to209gpm", "SOccDPT_{V2}"),


    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "yrd9xkse", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "oxdknlzn", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "tmgbbmtl", "SOccDPT_{V3}"),
    ("pw22-sbn-01", "SOcc_dpt_swin2_tiny_256", "ll84baru", "SOccDPT_{V3}"),


]

params_range = {
    # '_step': (0, 0),
    # 'epoch': (0, 0),
    # 'learning rate': (0, 0),
    # 'abs_rel': (0, 0),
    # 'sq_rel': (0, 0),
    # 'loss': (5.0, 35.0),
    'rmse': (12.0, 14.0),
    # 'rmse_log': (0, 0),
    'a1': (0.65, 0.75),
    'a2': (0.0, 1.0),
    'a3': (0.0, 1.0),
    'iou': (0.0, 1.0),

}


for p in params_range:
    fig = plt.figure(figsize=(10, 6))
    for entity, project, run_id, display_name in runs_list:
        # x = data['_step'][display_name]
        # y = data[p][display_name]

        # print(x)
        # print(y)

        # plt.plot(x, y, label=f"${display_name}$")
        x = data[p][display_name]['_step']
        y = data[p][display_name][p]

        # x in range (0, 700)
        x_mask = (x >= 0) & (x <= 15377)
        x = x[x_mask]
        y = y[x_mask]

        plt.plot(x, y, label=f"${display_name}$")

        print(display_name)

        print(data[p][display_name])

    print('-'*20)
    
    # log scale
    if p in [
        'learning rate',
    ]:
        plt.yscale('log')

    # plt.tight_layout()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.ylim(params_range[p][0], params_range[p][1])
    plt.xlabel('Step', fontsize=16)
    plt.ylabel(p, fontsize=16)
    plt.title(p, fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(f'media/wandb_plots/{p}.png', bbox_inches='tight')  # Saving the plot
    plt.show()

    plt.close(fig)
    plt.clf()
    plt.cla()
    plt.close()

