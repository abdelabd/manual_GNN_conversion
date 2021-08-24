import os
import yaml
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='build_hls_config.yml')
    add_arg('--directory', type=str, default="test")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)

    # check that local directory exists
    local_dir = os.path.abspath(args.directory)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError()

    # local<-->remote path-handling
    remote_dir = local_dir.replace(os.getcwd(), config['ssh_top_dir'])
    local_parent_dir = os.path.abspath(os.path.join(args.directory, os.pardir))
    remote_parent_dir = local_parent_dir.replace(os.getcwd(), config['ssh_top_dir'])
    config.update({"remote_dir": remote_dir, "remote_parent_dir": remote_parent_dir, "local_dir": local_dir})

    # write .sh file
    with open(os.getcwd() + "/build_hls_template.sh", "r") as f:
        sh_script = f.read()
    sh_script = sh_script.format(**config)
    with open(local_dir + "/build_hls.sh", "w") as f:
        f.write(sh_script)

    # run .sh file
    tic = time.time()
    print(f"Building project: {local_dir}")
    ret_val = os.system(f"bash {local_dir}/build_hls.sh")
    print(f"ret_val: {ret_val}")
    duration = time.time() - tic
    print(f"Duration: {duration//60} minutes, {duration%60} seconds")

if __name__ == "__main__":
    main()

