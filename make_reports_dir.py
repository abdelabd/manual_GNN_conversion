import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--dir', type=str, default='hls_output/n28xe51')
    return parser.parse_args()

def main():
    args = parse_args()
    all_project_names = [i for i in os.listdir(args.dir) if (i[-3:] != ".gz" and i[-4:] != ".csv")]
    all_project_names = [i for i in all_project_names if i not in ["summary.csv", "scan_plots", "reports"]]
    os.makedirs(os.path.join(args.dir, "reports"), exist_ok=True)

    for project in all_project_names:
        project_dest_dir = os.path.join(args.dir, "reports", project)
        os.makedirs(project_dest_dir, exist_ok=True)

        vsynth_source = os.path.join(args.dir, project, "vivado_synth.rpt")
        vsynth_dest = os.path.join(project_dest_dir, "vivado_synth.rpt")
        shutil.copyfile(vsynth_source, vsynth_dest)

        csynth_source = os.path.join(args.dir, project, "myproject_csynth.rpt")
        csynth_dest = os.path.join(project_dest_dir, "myproject_csynth.rpt")
        shutil.copyfile(csynth_source, csynth_dest)

if __name__=="__main__":
    main()