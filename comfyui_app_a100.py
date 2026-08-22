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

def update_custom_nodes(nodes_dir: str):
    print(f"--- Memulai Update Otomatis Custom Nodes di: {nodes_dir} ---")

    if not os.path.exists(nodes_dir):
        print(f"Direktori {nodes_dir} tidak ditemukan. Skipping update.")
        return

    for item in os.listdir(nodes_dir):
        node_path = os.path.join(nodes_dir, item)
        git_dir = os.path.join(node_path, ".git")

        if os.path.isdir(node_path) and os.path.isdir(git_dir):
            try:
                print(f"Updating: {item}...")
                subprocess.run(["git", "config", "pull.ff", "only"], cwd=node_path, check=False)
                result = subprocess.run(["git", "pull"], cwd=node_path, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    if "Already up to date" in result.stdout:
                        print(f"{item}: Sudah versi terbaru.")
                    else:
                        print(f"{item}: Berhasil di-update!")
                else:
                    print(f"{item} Error: {result.stderr.strip()}")
            except Exception as e:
                print(f"Gagal update {item}: {e}")

    print("--- Selesai Update Custom Nodes ---")

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL)
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))

import modal

# Build image with ComfyUI installed to default location /root/comfy/ComfyUI
image = (
    # UPGRADE PYTHON: Menggunakan versi Python 3.13
    # untuk mengoptimalkan efisiensi kompilasi dan manajemen memori.
    modal.Image.from_registry("python:3.13-slim-bookworm")
    
    # Menambahkan gcc & g++ untuk berjaga-jaga building library C-extension 
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "gcc", "g++")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        
        # FIX CUDA WARNING: Instalasi PyTorch dengan cu130
        "uv pip install --system --compile-bytecode torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130",
        
        # FIX HF_TRANSFER: Gunakan versi terbaru HF hub yang terintegrasi dengan Xet backend
        "uv pip install --system --compile-bytecode huggingface_hub[cli] hf_xet",
        
        # Install ComfyUI to default location
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    
    # Menambahkan matrix-nio, PyOpenGL-accelerate, dan deepdiff ke list dependencies
    .run_commands([
        "pip install --upgrade ftfy accelerate einops diffusers sentencepiece sageattention onnx onnxruntime onnxruntime-gpu nvidia-ml-py av matrix-nio PyOpenGL-accelerate deepdiff 'aiohttp==3.14.1'"
    ])
)

# Install nodes to default ComfyUI location during build
image = image.run_commands([
    "comfy node install rgthree-comfy comfyui-impact-pack comfyui-impact-subpack ComfyUI-YOLO comfyui-inspire-pack comfyui_ipadapter_plus wlsh_nodes ComfyUI_Comfyroll_CustomNodes comfyui_essentials ComfyUI-GGUF"
])

# Git-based nodes baked into image at default ComfyUI location
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
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
    # Menggunakan GPU A100-80GB
    gpu=os.environ.get('MODAL_GPU_TYPE', 'A100-80GB'),
    volumes={DATA_ROOT: vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():
    # Check if volume is empty (first run)
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI from default location to volume...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            os.makedirs(DATA_BASE, exist_ok=True)

    # Fix detached HEAD and update ComfyUI backend
    print("Fixing git branch and updating ComfyUI backend to the latest version...")
    os.chdir(DATA_BASE)
    try:
        result = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run("git checkout -B master origin/master", shell=True, check=True)
        subprocess.run("git config pull.ff only", shell=True, check=True)
        subprocess.run("git pull --ff-only", shell=True, check=True)
    except Exception as e:
        print(f"Error during backend update: {e}")

    # Patch ComfyUI-TeaCache for compatibility with latest ComfyUI versions
    teacache_nodes = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-TeaCache", "nodes.py")
    if os.path.exists(teacache_nodes):
        try:
            with open(teacache_nodes, 'r') as f:
                content = f.read()
            target_import = "from comfy.ldm.lightricks.model import precompute_freqs_cis"
            if target_import in content:
                print("Auto-Patching ComfyUI-TeaCache...")
                patch_code = """
try:
    from comfy.ldm.lightricks.model import precompute_freqs_cis
except ImportError:
    import torch
    def precompute_freqs_cis(coords, dim, out_dtype):
        theta = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=coords.device, dtype=torch.float32) / dim))
        freqs = torch.einsum('... , d -> ... d', coords.flatten(1, -2).to(torch.float32), theta)
        freqs = freqs.view(*coords.shape[:-1], -1)
        return torch.polar(torch.ones_like(freqs), freqs.to(out_dtype))
"""
                content = content.replace(target_import, patch_code.strip())
                with open(teacache_nodes, 'w') as f:
                    f.write(content)
        except Exception as e:
            print(f"Error patching TeaCache: {e}")

    # Configure built-in ComfyUI-Manager
    manager_config_dir = os.path.join(DATA_BASE, "user", "__manager")
    manager_config_path = os.path.join(manager_config_dir, "config.ini")
    legacy_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")

    if os.path.exists(legacy_dir):
        print("Migrating Manager data from legacy path...")
        os.makedirs(manager_config_dir, exist_ok=True)
        shutil.copytree(legacy_dir, manager_config_dir, dirs_exist_ok=True)
        shutil.rmtree(legacy_dir)

    backup_dir = os.path.join(manager_config_dir, ".legacy-manager-backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    os.makedirs(manager_config_dir, exist_ok=True)
    config_content = "[default]\nnetwork_mode = personal_cloud\nsecurity_level = weak\nlog_to_file = false\nauto_fetch = false\n"
    with open(manager_config_path, "w") as f:
        f.write(config_content)

    # FIX BLOCKED BY POLICY: Hapus folder lama ComfyUI-Manager di custom_nodes agar 
    # terhindar dari bentrok dengan ComfyUI-Manager Native
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        print("Menghapus custom_nodes/ComfyUI-Manager lama agar tidak bentrok dengan Manager Native...")
        try:
            shutil.rmtree(manager_dir)
        except Exception as e:
            print(f"Error menghapus manager lama: {e}")
            
    # Update ALL other custom nodes automatically
    update_custom_nodes(CUSTOM_NODES_DIR) 
    
    # Upgrade pip & comfy-cli at runtime
    try:
        subprocess.run("pip install --no-cache-dir --upgrade pip comfy-cli", shell=True, check=True)
    except Exception as e:
        print(f"Error upgrading pip/comfy-cli: {e}")

    # Update ComfyUI frontend by installing requirements
    requirements_path = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(requirements_path):
        subprocess.run(f"/usr/local/bin/python -m pip install -r {requirements_path}", shell=True, check=False)

    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # Download models at runtime
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            try:
                hf_download(sub, fn, repo, subf)
            except Exception as e:
                print(f"Error downloading {fn}: {e}")

    # Run extra download commands
    for cmd in extra_cmds:
        subprocess.run(cmd, shell=True, check=False, cwd=DATA_BASE)

    os.environ["COMFY_DIR"] = DATA_BASE

    # Launch ComfyUI (argumen --enable-manager akan mengaktifkan manager native tanpa konflik)
    print(f"Starting ComfyUI from {DATA_BASE}...")
    cmd = ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000", "--enable-manager"]
    process = subprocess.Popen(cmd, cwd=DATA_BASE, env=os.environ.copy())