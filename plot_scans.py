import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv(os.getcwd()+"/numbers_for_paper/fpga_part_xcvu9p.csv")
    
    precision_indeces = [i for i in range(df.shape[0]) if df["Project"].iloc[i][:2]=="ap"]
    df_precision = df.iloc[precision_indeces]
    df_precision["Project"] = [i.replace("ap_fixed<", "") for i in df_precision["Project"]]
    df_precision["Project"] = [i.replace(">", "") for i in df_precision["Project"]]
    
    reuse_indeces = [i for i in range(df.shape[0]) if i not in precision_indeces]
    df_reuse = df.iloc[reuse_indeces]
    df_reuse["Project"] = [i.replace("rf", "") for i in df_reuse["Project"]]
    
    project_names = list(df["Project"])
    var_names = list(df.columns)
    del var_names[0]
    
    for vname in var_names:
        plt.figure()
        plt.plot(df_precision["Project"], df_precision[vname])
        plt.xlabel("Fixed-point precision")
        plt.ylabel(vname)
        plt.title(f"{vname} by precision")
        plt.savefig(os.getcwd()+f"/numbers_for_paper/scan_plots/{vname} by precision.jpg")
        plt.show()
        plt.close()
        
        plt.figure()
        plt.plot(df_reuse["Project"], df_reuse[vname])
        plt.xlabel("Reuse Factor")
        plt.ylabel(vname)
        plt.title(f"{vname} by reuse-factor")
        plt.savefig(os.getcwd()+f"/numbers_for_paper/scan_plots/{vname} by reuse-factor.jpg")
        plt.show()
        plt.close()

    return df, project_names, var_names

if __name__=="__main__":
    df, project_names, var_names=main()