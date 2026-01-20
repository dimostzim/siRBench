import os
import subprocess

TOOL_CHOICES = [
    "oligoformer",
    "gnn4sirna",
    "sirnadiscovery",
    "attsioff",
    "sirnabert",
    "ensirna",
]


def tool_dir(base_dir, tool):
    return os.path.join(base_dir, "tools", tool)


def repo_root(base_dir):
    return os.path.abspath(os.path.join(base_dir, ".."))


def to_container_path(path, host_root, container_root="/work"):
    if not path:
        return path
    abs_path = os.path.abspath(path)
    host_root = os.path.abspath(host_root)
    if abs_path.startswith(host_root + os.sep):
        rel = os.path.relpath(abs_path, host_root)
        return os.path.join(container_root, rel)
    return path


def run_docker(tool, script_rel, argv, host_root, gpus="all"):
    image = f"{tool}:latest"
    workdir = f"/work/competitors/tools/{tool}"
    torch_home = os.environ.get("TORCH_HOME")
    rosetta_dir = os.environ.get("ROSETTA_DIR")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_root}:/work",
        "-w", workdir,
    ]
    if torch_home:
        cmd.extend(["-e", f"TORCH_HOME={torch_home}"])
    if rosetta_dir and os.path.isdir(rosetta_dir):
        if not os.path.abspath(rosetta_dir).startswith(os.path.abspath(host_root) + os.sep):
            cmd.extend(["-v", f"{rosetta_dir}:{rosetta_dir}"])
        cmd.extend(["-e", f"ROSETTA_DIR={rosetta_dir}"])
    if gpus:
        cmd.extend(["--gpus", gpus])
    cmd.append(image)
    cmd.extend(["python3", script_rel])
    cmd.extend(argv)
    subprocess.check_call(cmd)
