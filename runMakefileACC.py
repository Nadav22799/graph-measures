import os
import subprocess

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", None)


def makefile_command(gpu: bool):
    if not gpu:
        return ["conda run -n boost make -f Makefile-conda"]
    return ["conda run -n boost make -f Makefile-conda", "conda run -n boost make -f Makefile-gpu"]


def run_makefile():
    # if user use conda run makefile
    if CONDA_PREFIX is not None:
        print("Download version with accelerated calculation.")

        # check if gpu exists
        import torch
        if not torch.cuda.is_available():
            print("Does not support GPU.\nBuild accelerated features.")
            gpu_available = False
            # path = os.getcwd() + "/runMakefileCPU.py"
            # path = "os.system('make -f Makefile-conda')"
        else:
            print("Support GPU.\nBuild accelerated features for GPU.")
            gpu_available = True
            # path = os.getcwd() + "/runMakefileGPU.py"
            # path = "os.system('make -f Makefile-conda'); os.system('make -f Makefile-gpu')"

        # build Makefile
        os.chdir("graphMeasures/features_algorithms/accelerated_graph_features/src")
        print("cd ed")
        os.system("conda env create -f env.yml --force")
        print("created conda env")

        def conda_base():
            split_path = list(os.path.split(CONDA_PREFIX))
            while "conda" not in split_path[-1]:
                split_path = list(os.path.split(split_path[0]))
            return "/".join(split_path)

        cmd = '. ' + conda_base() + '/etc/profile.d/conda.sh && conda activate boost'
        print(cmd)
        subprocess.call(cmd, shell=True, executable='/bin/bash')
        
        print("did subprocess thingy")
        for command in makefile_command(gpu_available):
            process = subprocess.Popen(
                command.split(), stdout=subprocess.PIPE)

            print("another subprocessing")
            output = process.stdout.read()
            print(output)
            output, error = process.communicate()
            print(output)
            print(error)

    else:
        print("Does not use Conda environment or Linux.\nDownload version without accelerated calculation.")


if __name__ == '__main__':
    run_makefile()
