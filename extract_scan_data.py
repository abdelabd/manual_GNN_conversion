import os
import yaml
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dir', type=str, default='hls_output/n28xe51_manual_fpib')
    return parser.parse_args()

def parse_vsynth_report(dir):
    with open(dir, "r") as f:
        file=f.read()
    file = file.split("\n")

    terms_of_interest = ["CLB LUTs*", "CARRY8", "DSPs", "Bonded IOB", "RAMB18"]
    percent_util_dict = {i: None for i in terms_of_interest}
    for line in file:
        for term, val in percent_util_dict.items():
            if (term in line) and val==None:
                used = float(line.split("|")[2].strip())
                available = float(line.split("|")[4].strip())
                percent_util_dict[term] = 100*used/available

    out_dict = {k+" (%) V-Synth":v for k,v in percent_util_dict.items()}

    for line in file:
        if "Register as Flip Flop" in line:
            used = float(line.split("|")[2].strip())
            available = float(line.split("|")[4].strip())
            out_dict["FF (%) V-Synth"] = 100*used/available
    return out_dict

def parse_csynth_report(dir):
    with open(dir, "r") as f:
        file = f.read()
    file = file.split("\n")

    out_dict = {}
    latency_found = False
    usage_found = False

    for i, line in enumerate(file):
        if not latency_found:
            if ("Latency   |  Interval" in line) or ("Latency  |  Interval" in line):
                line_of_interest = file[i+3].split("|")
                out_dict["Latency min (clock cycles)"] = line_of_interest[1]
                out_dict["Latency max (clock cycles)"] = line_of_interest[2]
                out_dict["Interval min (clock cycles)"] = line_of_interest[3]
                out_dict["Interval max (clock cycles)"] = line_of_interest[4]
                latency_found = True

        if ("|Total                |" in line) and not usage_found:
            line_of_interest = line.split("|")
            bram18k_used = float(line_of_interest[2].strip())
            dsp48_used = float(line_of_interest[3].strip())
            ff_used = float(line_of_interest[4].strip())
            lut_used = float(line_of_interest[5].strip())
            uram_used = float(line_of_interest[6].strip())

            try:
                available_slr_line = file[i+2]
            except IndexError:
                continue

            line_of_interest = available_slr_line.split("|")
            bram18k_slr_avail = float(line_of_interest[2].strip())
            dsp48_slr_avail= float(line_of_interest[3].strip())
            ff_slr_avail = float(line_of_interest[4].strip())
            lut_slr_avail = float(line_of_interest[5].strip())
            uram_slr_avail = float(line_of_interest[6].strip())

            out_dict["BRAM_18K SLR (%) C-Synth"] = 100*bram18k_used / bram18k_slr_avail
            out_dict["DSP48E SLR (%) C-Synth"] = 100*dsp48_used / dsp48_slr_avail
            out_dict["FF SLR (%) C-Synth"] = 100*ff_used / ff_slr_avail
            out_dict["LUT SLR (%) C-Synth"] = 100*lut_used / lut_slr_avail
            out_dict["URAM SLR (%) C-Synth"] = 100*uram_used / uram_slr_avail

            try:
                available_line = file[i+6]
            except IndexError:
                continue

            line_of_interest = available_line.split("|")
            bram18k_avail = float(line_of_interest[2].strip())
            dsp48_avail = float(line_of_interest[3].strip())
            ff_avail = float(line_of_interest[4].strip())
            lut_avail = float(line_of_interest[5].strip())
            uram_avail = float(line_of_interest[6].strip())

            out_dict["BRAM_18K (%) C-Synth"] = 100*bram18k_used/bram18k_avail
            out_dict["DSP48E (%) C-Synth"] = 100*dsp48_used/dsp48_avail
            out_dict["FF (%) C-Synth"] = 100*ff_used/ff_avail
            out_dict["LUT (%) C-Synth"] = 100*lut_used/lut_avail
            out_dict["URAM (%) C-Synth"] = 100*uram_used/uram_avail

            usage_found = True

    return out_dict

def get_single_report(dir):
    vsynth_dict = parse_vsynth_report(dir+"/vivado_synth.rpt")
    csynth_dict = parse_csynth_report(dir+"/myproject_csynth.rpt")
    vsynth_dict.update(csynth_dict)
    return vsynth_dict

def main():
    args = parse_args()

    columns = ["Project", "CLB LUTs* (%) V-Synth", "CARRY8 (%) V-Synth", "DSPs (%) V-Synth", "Bonded IOB (%) V-Synth",
               "RAM18 (%) V-Synth",
               "Latency min (clock cycles)", "Latency max (clock cycles)", "Interval min (clock cycles)",
               "Interval max (clock cycles)", "BRAM_18K SLR (%) C-Synth", "DSP48E SLR (%) C-Synth",
               "FF SLR (%) C-Synth", "LUT SLR (%) C-Synth", "URAM SLR (%) C-Synth", "BRAM_18K (%) C-Synth",
               "DSP48E (%) C-Synth", "FF (%) C-Synth", "LUT (%) C-Synth", "URAM (%) C-Synth"]
    df = pd.DataFrame(columns=columns)

    all_project_dirs = [i for i in os.listdir(args.dir) if (i[-3:]!=".gz" and i[-4:]!=".csv")]
    for i, project_dir in enumerate(all_project_dirs):
        out_dict = get_single_report(os.path.join(args.dir, project_dir))
        out_dict["Project"] = project_dir
        df = df.append(out_dict, ignore_index=True)

    df.to_csv(args.dir+"/summary.csv", index=False)

if __name__=="__main__":
    main()
