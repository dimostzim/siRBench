import os
import subprocess
import sys

QUIET = os.environ.get("QUIET", "1") == "1"

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


def run_docker(tool, script_rel, argv, host_root, status_msg=None):
    image = f"{tool}:latest"
    workdir = f"/work/competitors/tools/{tool}"
    torch_home = os.environ.get("TORCH_HOME")
    rosetta_dir = os.environ.get("ROSETTA_DIR")
    if not rosetta_dir and tool == "ensirna":
        default_rosetta = os.path.join(host_root, "competitors", "tools", "ensirna", "rosetta")
        if os.path.isdir(default_rosetta):
            rosetta_dir = default_rosetta
    uid = os.getuid() if hasattr(os, "getuid") else None
    gid = os.getgid() if hasattr(os, "getgid") else None
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_root}:/work",
        "-w", workdir,
        "--gpus", "all",
        "-e", "PYTHONWARNINGS=ignore::UserWarning",
        "-e", "HOME=/tmp",
        "-e", "XDG_CACHE_HOME=/tmp",
    ]
    if uid is not None and gid is not None:
        cmd.extend(["-u", f"{uid}:{gid}"])
    if torch_home:
        cmd.extend(["-e", f"TORCH_HOME={torch_home}"])
    if rosetta_dir and os.path.isdir(rosetta_dir):
        rosetta_dir_container = to_container_path(rosetta_dir, host_root)
        if not os.path.abspath(rosetta_dir).startswith(os.path.abspath(host_root) + os.sep):
            cmd.extend(["-v", f"{rosetta_dir}:{rosetta_dir}"])
        cmd.extend(["-e", f"ROSETTA_DIR={rosetta_dir_container}"])
    cmd.append(image)
    cmd.extend(["python3", script_rel])
    cmd.extend(argv)

    if QUIET:
        if status_msg:
            print(status_msg, end="... ", flush=True)
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if status_msg:
            print("done")
    else:
        subprocess.check_call(cmd)
