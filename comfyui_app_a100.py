import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"

# ComfyUI default install location
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"

def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False) -> str:
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # dependencies for WanVideoWrapper node
    .run_commands([
        "pip install ftfy accelerate einops diffusers sentencepiece"
    ])
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# Model download tasks (will be done at runtime)
model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# Create volume
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=1800,
    gpu=os.environ.get('MODAL_GPU_TYPE', 'A100-40GB'),
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)  # Increased timeout for handling restarts
def ui():
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        # Ensure DATA_ROOT exists
        os.makedirs(DATA_ROOT, exist_ok=True)
        # Copy ComfyUI from default location to volume
        if os.path.exists(DEFAULT_COMFY_DIR):
            print(f"Copying {DEFAULT_COMFY_DIR} to {DATA_BASE}")
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            print(f"Warning: {DEFAULT_COMFY_DIR} not found, creating empty structure")
            os.makedirs(DATA_BASE, exist_ok=True)

    # Fix detached HEAD and update ComfyUI backend to the latest version
    print("Fixing git branch and updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        # Check if in detached HEAD state
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Detected detached HEAD, checking out master branch...")
            subprocess.run("git checkout -B master origin/master", shell=True, check=True, capture_output=True, text=True)
            print("Successfully checked out master branch")
        # Configure pull strategy to fast-forward only
        subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
        # Perform git pull
        result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
        print("Git pull output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error updating ComfyUI backend: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during backend update: {e}")

    # Define paths for Manager config (use new secure path)
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    legacy_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")

    # Migrate from legacy path if it exists
    if os.path.exists(legacy_dir):
        print("Migrating Manager data from legacy path to __manager...")
        os.makedirs(manager_config_dir, exist_ok=True)
        shutil.copytree(legacy_dir, manager_config_dir, dirs_exist_ok=True)  # Copy contents
        shutil.rmtree(legacy_dir)  # Delete legacy dir to prevent detection
        print("Migration completed and legacy dir removed.")

    # Delete any legacy backup to stop persistent notifications
    backup_dir = os.path.join(manager_config_dir, ".legacy-manager-backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
        print(f"Removed legacy backup at {backup_dir} to stop notifications")

    # Configure ComfyUI-Manager: Disable auto-fetch, set weak security, and disable file logging
    print("Configuring ComfyUI-Manager: Disabling auto-fetch, setting security_level to weak, and disabling file logging...")
    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)
    print(f"Updated {manager_config_path} with security_level=weak, log_to_file=false")

    # Now update ComfyUI-Manager to the latest version (config already set, so should allow)
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Updating ComfyUI-Manager to the latest version...")
        os.chdir(manager_dir)
        try:
            # Configure pull strategy for ComfyUI-Manager
            subprocess.run("git config pull.ff only", shell=True, check=True, capture_output=True, text=True)
            result = subprocess.run("git pull --ff-only", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager git pull output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI-Manager: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during ComfyUI-Manager update: {e}")
        os.chdir(DATA_BASE)  # Return to base directory
    else:
        print("ComfyUI-Manager directory not found, installing...")
        try:
            subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=True, capture_output=True, text=True)
            print("ComfyUI-Manager installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing ComfyUI-Manager: {e.stderr}")

    # Upgrade pip at runtime
    print("Upgrading pip at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=True, capture_output=True, text=True)
        print("pip upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during pip upgrade: {e}")

    # Upgrade comfy-cli at runtime
    print("Upgrading comfy-cli at runtime...")
    try:
        result = subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=True, capture_output=True, text=True)
        print("comfy-cli upgrade output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading comfy-cli: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during comfy-cli upgrade: {e}")

    # Update ComfyUI frontend by installing requirements
    print("Updating ComfyUI frontend by installing requirements...")
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            result = subprocess.run(
                f"/usr/local/bin/python -m pip install -r {requirements_path}",
                shell=True, check=True, capture_output=True, text=True
            )
            print("Frontend update output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error updating ComfyUI frontend: {e.stderr}")
        except Exception as e:
            print(f"Unexpected error during frontend update: {e}")
    else:
        print(f"Warning: {requirements_path} not found, skipping frontend update")

    # Install pip dependencies for new ComfyUI Manager
    print("Installing pip dependencies for new ComfyUI Manager...")
    manager_req_path = os.path.join(DATA_BASE, "manager_requirements.txt")
    if os.path.exists(manager_req_path):
        try:
            result = subprocess.run(
                f"pip install -r {manager_req_path}",
                shell=True, check=True, capture_output=True, text=True
            )
            print("New Manager dependencies installed:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error installing new Manager dependencies: {e.stderr}")
    else:
        print(f"Warning: {manager_req_path} not found, skipping new Manager dependencies installation")

    # Ensure all required directories exist
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime (only if missing)
    print("Checking and downloading missing models...")
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            print(f"Downloading {fn} to {target}...")
            try:
                hf_download(sub, fn, repo, subf)
                print(f"Successfully downloaded {fn}")
            except Exception as e:
                print(f"Error downloading {fn}: {e}")
        else:
            print(f"Model {fn} already exists, skipping download")

    # Run extra download commands
    print("Running additional downloads...")
    for cmd in extra_cmds:
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Command completed successfully")
            else:
                print(f"Command failed with return code {result.returncode}: {result.stderr}")
        except Exception as e:
            print(f"Error running command {cmd}: {e}")

    # Set COMFY_DIR environment variable to volume location
    os.environ["COMFY_DIR"] = DATA_BASE

    # Launch ComfyUI from volume location
    print(f"Starting ComfyUI from {DATA_BASE}...")
    # Start ComfyUI server with correct syntax and latest frontend
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest", "--enable-manager"]
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, cwd=DATA_BASE, env=os.environ.copy()
    )