import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dir', type=str, default='hls_output/n28xe51_manual_fpib')
    add_arg('--show-plots', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.dir+"/summary.csv")
    df.drop(columns = ["URAM SLR (%) C-Synth", "URAM (%) C-Synth", "Latency min (clock cycles)", "Interval min (clock cycles)"], inplace=True)
    df.rename(columns={"Latency max (clock cycles)": "Latency (clock cycles)", "Interval max (clock cycles)": "Interval (clock cycles)"}, inplace=True)

    precision_indeces = [i for i in range(df.shape[0]) if df["Project"].iloc[i][:2]=="ap"]
    df_precision = df.iloc[precision_indeces].copy()
    df_precision["Project"] = [i.replace("ap_fixed_", "") for i in df_precision["Project"]]
    df_precision["Project"] = [i.replace("_", ",") for i in df_precision["Project"]]
    df_precision["fpb"] = [int(i[0]) for i in df_precision["Project"].str.split(",")]
    df_precision.sort_values("fpb", ascending=True, inplace=True)

    reuse_indeces = [i for i in range(df.shape[0]) if i not in precision_indeces]
    df_reuse = df.iloc[reuse_indeces].copy()
    df_reuse["Project"] = [int(i.replace("rf", "")) for i in df_reuse["Project"]]
    df_reuse.sort_values("Project", ascending=True, inplace=True)
    
    project_names = list(df["Project"])
    var_names = list(df.columns)
    del var_names[0]

    os.makedirs(args.dir + "/scan_plots", exist_ok=True)
    for vname in var_names:
        plt.figure()
        plt.plot(df_precision["Project"], df_precision[vname])
        plt.xlabel("Fixed-point precision")
        plt.ylabel(vname)
        plt.title(f"{vname} by precision")
        plt.savefig(args.dir+f"/scan_plots/{vname} by precision.jpg")
        if args.show_plots:
            plt.show()
        plt.close()
        
        plt.figure()
        plt.plot(df_reuse["Project"], df_reuse[vname])
        plt.xlabel("Reuse Factor")
        plt.ylabel(vname)
        plt.xticks(ticks = df_reuse["Project"], labels=["%s"%i for i in df_reuse["Project"]])
        plt.title(f"{vname} by reuse-factor")
        plt.savefig(args.dir+f"/scan_plots/{vname} by reuse-factor.jpg")
        if args.show_plots:
            plt.show()
        plt.close()

if __name__=="__main__":
    main()