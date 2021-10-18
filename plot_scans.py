import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dir', type=str, default='hls_output/n28xe51_manual_fpib')
    add_arg('--max-nodes', type=int, default=28)
    add_arg('--max-edges', type=int, default=51)
    add_arg('--reuse', type=int, default=8)
    add_arg('--precision', type=str, default="ap_fixed<16,8>")
    add_arg('--show-plots', action='store_true')
    return parser.parse_args()

def main():
    
    
    args = parse_args()
    df = pd.read_csv(args.dir+"/summary.csv")
    df.drop(columns = ["URAM SLR (%) C-Synth", "URAM (%) C-Synth", "Latency min (clock cycles)", 
                       "Interval min (clock cycles)", "FF SLR (%) C-Synth", "Bonded IOB (%) V-Synth",
                       "BRAM_18K SLR (%) C-Synth", "DSP48E SLR (%) C-Synth", "CARRY8 (%) V-Synth"], inplace=True)
    df.rename(columns={"Latency max (clock cycles)": "Latency (clock cycles)", 
                       "Interval max (clock cycles)": "II (clock cycles)"}, inplace=True)

    precision_indeces = [i for i in range(df.shape[0]) if df["Project"].iloc[i][:2]=="ap"]
    df_precision = df.iloc[precision_indeces].copy()
    df_precision["Project"] = [i.replace("ap_fixed_", "") for i in df_precision["Project"]]
    df_precision["Project"] = [i.replace("_", ",") for i in df_precision["Project"]]
    df_precision["fpb"] = df_precision["Project"].str.split(',', n = 1, expand = True)[0].astype(int)
    df_precision.sort_values("fpb", ascending=True, inplace=True)
    df_precision.replace('    ~0   ', 0, inplace=True)
    
    reuse_indeces = [i for i in range(df.shape[0]) if i not in precision_indeces]
    df_reuse = df.iloc[reuse_indeces].copy()
    df_reuse["Project"] = [int(i.replace("rf", "")) for i in df_reuse["Project"]]
    df_reuse.sort_values("Project", ascending=True, inplace=True)
    df_reuse.replace('    ~0   ', 0, inplace=True)

    project_names = list(df["Project"])
    var_names = list(df.columns)
    del var_names[0]

    os.makedirs(args.dir + "/scan_plots", exist_ok=True)
    
    csynth_vars = ['LUT (%) C-Synth', 'DSP48E (%) C-Synth', 'FF (%) C-Synth', 'BRAM_18K (%) C-Synth']
    
    csynth_legend_names = {'BRAM_18K (%) C-Synth': 'BRAM', 
                           'DSP48E (%) C-Synth': 'DSP', 
                           'FF (%) C-Synth': 'FF', 
                           'LUT (%) C-Synth': 'LUT'}
    
    vsynth_vars = ['CLB LUTs* (%) V-Synth', 'DSPs (%) V-Synth', 'FF (%) V-Synth', 'RAMB18 (%) V-Synth']
    
    vsynth_legend_names = {'RAMB18 (%) V-Synth': 'BRAM',
                           'DSPs (%) V-Synth': 'DSP', 
                           'FF (%) V-Synth': 'FF', 
                           'CLB LUTs* (%) V-Synth': 'LUT'}
    plt.figure()
    for vname in csynth_vars:
        plt.plot(df_precision["fpb"], df_precision[vname].astype(int), label=csynth_legend_names[vname],  ls='-', marker='o', lw=2)
    plt.xlabel("Total bits")
    plt.ylabel("Resource usage [%]")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, C synth.\nRF = {args.reuse}")
    plt.savefig(args.dir+"/scan_plots/precision_scan_resource_csynth.pdf")
    plt.savefig(args.dir+"/scan_plots/precision_scan_resource_csynth.png")
    if args.show_plots:
        plt.show()
    plt.close()
    
    plt.figure()
    for vname in vsynth_vars:
        plt.plot(df_precision["fpb"], df_precision[vname].astype(int), label=vsynth_legend_names[vname],  ls='-', marker='o', lw=2)
    plt.xlabel("Total bits")
    plt.ylabel("Resource usage [%]")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, logic synth.\nRF = {args.reuse}")
    plt.savefig(args.dir+"/scan_plots/precision_scan_resource_vsynth.pdf")
    plt.savefig(args.dir+"/scan_plots/precision_scan_resource_vsynth.png")
    if args.show_plots:
        plt.show()
    plt.close()
        
    plt.figure()
    plt.plot(df_precision["fpb"], df_precision["Latency (clock cycles)"], label="Latency",  ls='-', marker='o', lw=2)
    plt.plot(df_precision["fpb"], df_precision["II (clock cycles)"], label="II",  ls='-', marker='o', lw=2)
    plt.xlabel("Total bits")
    plt.ylabel("Clock cycles")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, C synth.\nRF = {args.reuse}")
    plt.savefig(args.dir+"/scan_plots/precision_scan_time_csynth.pdf")
    plt.savefig(args.dir+"/scan_plots/precision_scan_time_csynth.png")
    if args.show_plots:
        plt.show()
    plt.close()
    
    plt.figure()
    for vname in csynth_vars:
        plt.plot(df_reuse["Project"], df_reuse[vname].astype(int), label=csynth_legend_names[vname], ls='-', marker='o', lw=2)
    plt.xlabel("Reuse factor")
    plt.ylabel("Resource usage [%]")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, C synth.\n{args.precision}")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_resource_csynth.pdf")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_resource_csynth.png")
    if args.show_plots:
        plt.show()
    plt.close()
    
    plt.figure()
    for vname in vsynth_vars:
        plt.plot(df_reuse["Project"], df_reuse[vname].astype(int), label=vsynth_legend_names[vname], ls='-', marker='o', lw=2)
    plt.xlabel("Reuse factor")
    plt.ylabel("Resource usage [%]")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, logic synth.\n{args.precision}")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_resource_vsynth.pdf")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_resource_vsynth.png")
    if args.show_plots:
        plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(df_reuse["Project"], df_reuse["Latency (clock cycles)"], label="Latency",  ls='-', marker='o', lw=2)
    plt.plot(df_reuse["Project"], df_reuse["II (clock cycles)"], label="II",  ls='-', marker='o', lw=2)
    plt.xlabel("Reuse factor")
    plt.ylabel("Clock cycles")
    plt.legend(title=f"{args.max_nodes} nodes, {args.max_edges} edges, C synth.\n{args.precision}")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_time_csynth.pdf")
    plt.savefig(args.dir+"/scan_plots/reuse_factor_scan_time_csynth.png")
    if args.show_plots:
        plt.show()
    plt.close()

if __name__=="__main__":
    main()
