import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dir', type=str, default='hls_output/size_scan')
    add_arg('--precision', type=str, default="ap_fixed<14,7>")
    add_arg('--reuse', type=int, default=8)
    add_arg('--show-plots', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dir + "/scan_plots", exist_ok=True)

    # get data
    df = pd.read_csv(args.dir + "/summary.csv")
    df.drop(columns=["URAM SLR (%) C-Synth", "URAM (%) C-Synth", "Latency min (clock cycles)",
                     "Interval min (clock cycles)", "FF SLR (%) C-Synth", "Bonded IOB (%) V-Synth",
                     "BRAM_18K SLR (%) C-Synth", "DSP48E SLR (%) C-Synth", "CARRY8 (%) V-Synth"], inplace=True)
    df.rename(columns={"Latency max (clock cycles)": "Latency (clock cycles)",
                       "Interval max (clock cycles)": "II (clock cycles)"}, inplace=True)

    # define csynth variables, vsynth variables
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

    project_names = list(df["Project"])
    var_names = list(df.columns)
    del var_names[0]

    display = {}
    display['pipeline'] = 'Throughput-optimized'
    display['dataflow'] = 'Resource-optimized'
    # plot each paradigm separately
    for paradigm in ["pipeline", "dataflow"]:

        # get data from just that paradigm
        paradigm_indeces = [i for i in range(df.shape[0]) if paradigm in df["Project"].iloc[i]]
        if len(paradigm_indeces)==0:
            print(f"No {paradigm} projects found, check project names")
            continue

        df_paradigm = df.iloc[paradigm_indeces].copy()
        df_paradigm["Project"] = [i.replace(f"_{paradigm}", "") for i in df_paradigm["Project"]]

        # order by n_nodes
        n_nodes = np.zeros(df_paradigm.shape[0])
        for i, pname in enumerate(df_paradigm["Project"]):
            n_node_i = int(re.findall(r"n[0-999]+", pname)[0][1:])
            n_nodes[i] = n_node_i
        df_paradigm["n_nodes"] = n_nodes
        df_paradigm.sort_values("n_nodes", ascending=True, inplace=True)

        # plot csynth resource-usage
        plt.figure()
        for vname in csynth_vars:
            plt.plot(df_paradigm["n_nodes"], df_paradigm[vname].astype(int), label=csynth_legend_names[vname], ls='-',
                     marker='o', lw=4, ms=10)
        plt.xlabel("Number of nodes")
        plt.ylabel("Resource usage [%]")
        plt.legend(title=f"{display[paradigm]}, C synth.\n{args.precision}\nRF = {args.reuse}")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_resource_csynth.pdf")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_resource_csynth.png")
        if args.show_plots:
            plt.show()
        plt.close()

        # plot vsynth resource-usage
        plt.figure()
        for vname in vsynth_vars:
            plt.plot(df_paradigm["n_nodes"], df_paradigm[vname].astype(int), label=vsynth_legend_names[vname], ls='-',
                     marker='o', lw=4, ms=10)
        plt.xlabel("Number of nodes")
        plt.ylabel("Resource usage [%]")
        plt.legend(title=f"{display[paradigm]}, logic synth.\n{args.precision}\nRF = {args.reuse}")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_resource_vsynth.pdf")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_resource_vsynth.png")
        if args.show_plots:
            plt.show()
        plt.close()

        # plot csynth latency/II
        plt.figure()
        plt.plot(df_paradigm["n_nodes"], df_paradigm["Latency (clock cycles)"], label="Latency", ls='-', marker='o', lw=4, ms=10)
        plt.plot(df_paradigm["n_nodes"], df_paradigm["II (clock cycles)"], label="II", ls='-', marker='o', lw=4, ms=10)
        plt.xlabel("Number of nodes")
        plt.ylabel("Clock cycles")
        plt.legend(title=f"{display[paradigm]}, C synth.\n{args.precision}\nRF = {args.reuse}")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_time_csynth.pdf")
        plt.savefig(args.dir + f"/scan_plots/{paradigm}_time_csynth.png")
        if args.show_plots:
            plt.show()
        plt.close()

if __name__ == "__main__":
    main()
