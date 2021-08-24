#! /bin/bash

ssh {ssh_name} "rm -rf {remote_dir}; mkdir -p {remote_parent_dir}; exit"
scp -r {local_dir} {ssh_name}:{remote_dir}

ssh {ssh_name} "{remote_vivado_source}; cd {remote_dir}; vivado_hls build_prj.tcl; exit"

scp -r {ssh_name}:{remote_dir} {local_dir}
scp {ssh_name}:{remote_dir}/myproject_prj/solution1/syn/report/myproject_csynth.rpt {local_dir}/myproject_csynth.rpt
scp {ssh_name}:{remote_dir}/vivado_synth.rpt {local_dir}/vivado_synth.rpt
ssh {ssh_name} "rm -rf {remote_dir}; exit"
