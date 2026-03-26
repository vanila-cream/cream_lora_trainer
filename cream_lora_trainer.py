"""
Cream LoRA Trainer Node for ComfyUI

Trains SDXL LoRAs using kohya-ss/sd-scripts.
Based on Realtime Lora Trainer's sdxl_lora_trainer.py,
simplified for dataset-folder-based training.
"""

import os
import sys
import json
import hashlib
import queue
import re
import shutil
import subprocess
import tempfile
import threading

import folder_paths

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

from .cream_config_template import (
    generate_training_config,
    save_config,
    VRAM_PRESETS,
)


def _get_venv_python_path(sd_scripts_path):
    """Get the Python path for sd-scripts venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            python_path = os.path.join(sd_scripts_path, venv_folder, "Scripts", "python.exe")
        else:
            python_path = os.path.join(sd_scripts_path, venv_folder, "bin", "python")

        if os.path.exists(python_path):
            return python_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(sd_scripts_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(sd_scripts_path, "venv", "bin", "python")


def _get_accelerate_path(sd_scripts_path):
    """Get the accelerate path for sd-scripts venv based on platform.
    Checks both .venv (uv default) and venv (traditional) folders."""
    venv_folders = [".venv", "venv"]

    for venv_folder in venv_folders:
        if sys.platform == 'win32':
            accel_path = os.path.join(sd_scripts_path, venv_folder, "Scripts", "accelerate.exe")
        else:
            accel_path = os.path.join(sd_scripts_path, venv_folder, "bin", "accelerate")

        if os.path.exists(accel_path):
            return accel_path

    # Return traditional path for error messaging
    if sys.platform == 'win32':
        return os.path.join(sd_scripts_path, "venv", "Scripts", "accelerate.exe")
    else:
        return os.path.join(sd_scripts_path, "venv", "bin", "accelerate")


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MIXED_PRECISION_PATTERN = re.compile(r'^\s*mixed_precision\s*=\s*"([^"]+)"\s*$', re.MULTILINE)

# Default installation directory for sd-scripts (alongside this custom node)
DEFAULT_SD_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "sd-scripts")
SETUP_VERSION = "1"  # Bump to force re-install of dependencies


def _find_uv():
    """Find the uv executable. Searches env vars, ComfyUI paths, PATH."""
    import sysconfig
    import shutil as _shutil

    # Check environment variables (ComfyUI Desktop sets these)
    for env_key in ("COMFYUI_UV", "COMFY_DESKTOP_UV", "UV"):
        candidate = os.environ.get(env_key)
        if candidate and os.path.exists(candidate):
            return candidate

    # Check relative to sys.executable (ComfyUI bundled uv)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    if sys.platform == 'win32':
        relative_candidates = [
            "uv.exe",
            os.path.join("Scripts", "uv.exe"),
            os.path.join("..", "resources", "uv", "win", "uv.exe"),
            os.path.join("..", "..", "resources", "uv", "win", "uv.exe"),
        ]
    else:
        relative_candidates = [
            "uv",
            os.path.join("bin", "uv"),
        ]
    for rel in relative_candidates:
        candidate = os.path.normpath(os.path.join(exe_dir, rel))
        if os.path.exists(candidate):
            return candidate

    # Check Python scripts directory
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        uv_name = "uv.exe" if sys.platform == 'win32' else "uv"
        candidate = os.path.join(scripts_dir, uv_name)
        if os.path.exists(candidate):
            return candidate

    # Check PATH
    return _shutil.which("uv")


def _ensure_uv():
    """Find uv or install it via pip. Returns the uv executable path."""
    uv = _find_uv()
    if uv is not None:
        return uv

    # Install uv via pip
    print("[Cream LoRA] uv 설치 중...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "uv"],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"uv 설치 실패:\n{result.stderr.strip()[-2000:]}\n"
            f"수동 설치: pip install uv"
        )

    uv = _find_uv()
    if uv is None:
        raise RuntimeError("uv 설치 후에도 찾을 수 없습니다. pip install uv로 직접 설치해주세요.")
    print(f"[Cream LoRA] uv 설치 완료: {uv}")
    return uv




def _ensure_sd_scripts():
    """Ensure sd-scripts is installed. Download and set up if needed.
    Uses uv for fast, reliable dependency installation (following
    comfyui-instant-reference pattern).
    Returns the path to the sd-scripts directory.
    """
    sd_scripts_path = DEFAULT_SD_SCRIPTS_DIR
    train_script = os.path.join(sd_scripts_path, "sdxl_train_network.py")
    venv_dir = os.path.join(sd_scripts_path, "venv")
    marker = os.path.join(venv_dir, ".sd_scripts_ready")

    # ── Already fully set up? ────────────────────────────────────
    if os.path.exists(train_script) and os.path.exists(marker):
        try:
            with open(marker, 'r', encoding='utf-8') as f:
                if f.read().strip() == SETUP_VERSION:
                    accelerate_path = _get_accelerate_path(sd_scripts_path)
                    if os.path.exists(accelerate_path):
                        print(f"[Cream LoRA] 기존 sd-scripts 사용: {sd_scripts_path}")
                        return sd_scripts_path
        except Exception:
            pass

    # ── Check git availability ────────────────────────────────────
    try:
        subprocess.run(
            ["git", "--version"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise EnvironmentError(
            "sd-scripts 자동 설치에 git이 필요합니다.\n"
            "git을 설치하거나, sd_scripts_path를 직접 지정해주세요.\n"
            "git 다운로드: https://git-scm.com/downloads"
        )

    # ── Clone sd-scripts ──────────────────────────────────────────
    if not os.path.exists(train_script):
        print(f"[Cream LoRA] sd-scripts 다운로드 중... (최초 1회)")
        result = subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/kohya-ss/sd-scripts.git",
                sd_scripts_path,
            ],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sd-scripts 다운로드 실패:\n{result.stderr.strip()}"
            )
        print(f"[Cream LoRA] sd-scripts 다운로드 완료: {sd_scripts_path}")

    # ── Ensure uv is available ───────────────────────────────────
    uv = _ensure_uv()

    # ── Create venv using uv (with managed Python 3.12) ─────────
    if not os.path.exists(venv_dir):
        print(f"[Cream LoRA] 가상 환경 생성 중 (Python 3.12)...")
        result = subprocess.run(
            [uv, "venv", "--python", "3.12", venv_dir],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"가상 환경 생성 실패:\n{result.stderr.strip()}\n"
                f"uv가 Python 3.12를 자동 다운로드하지 못했습니다."
            )
        print(f"[Cream LoRA] 가상 환경 생성 완료")

    # ── Install dependencies with uv ─────────────────────────────
    if sys.platform == 'win32':
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    requirements_file = os.path.join(sd_scripts_path, "requirements.txt")
    if not os.path.exists(requirements_file):
        raise FileNotFoundError(
            f"requirements.txt를 찾을 수 없습니다: {requirements_file}"
        )

    print(f"[Cream LoRA] 의존성 설치 중... (첫 실행 시 5~10분 소요)")

    # Install requirements using uv pip
    result = subprocess.run(
        [
            uv, "pip", "install",
            "--python", venv_python,
            "-r", requirements_file,
        ],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=sd_scripts_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"의존성 설치 실패:\n{result.stderr.strip()[-2000:]}"
        )

    # Install PyTorch with CUDA support
    # sd-scripts' requirements.txt excludes torch/torchvision because
    # they require CUDA-specific wheels from a separate index.
    print(f"[Cream LoRA] PyTorch (CUDA) 설치 중...")
    pytorch_index = "https://download.pytorch.org/whl/cu124"
    result = subprocess.run(
        [
            uv, "pip", "install",
            "--python", venv_python,
            "--index-url", pytorch_index,
            "torch", "torchvision", "xformers",
        ],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=sd_scripts_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PyTorch 설치 실패:\n{result.stderr.strip()[-2000:]}"
        )
    print(f"[Cream LoRA] PyTorch 설치 완료")

    # Ensure accelerate is installed
    result = subprocess.run(
        [uv, "pip", "install", "--python", venv_python, "accelerate"],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        cwd=sd_scripts_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"accelerate 설치 실패:\n{result.stderr.strip()[-2000:]}"
        )

    print(f"[Cream LoRA] 의존성 설치 완료")

    # ── Final verification & marker ──────────────────────────────
    accelerate_path = _get_accelerate_path(sd_scripts_path)
    if not os.path.exists(accelerate_path):
        raise RuntimeError(
            f"설치가 완료되었지만 accelerate를 찾을 수 없습니다: {accelerate_path}\n"
            f"sd_scripts_path를 직접 지정해주세요."
        )

    # Write marker so we skip setup next time
    with open(marker, 'w', encoding='utf-8') as f:
        f.write(f"{SETUP_VERSION}\n")

    print(f"[Cream LoRA] sd-scripts 자동 설치 완료!")
    return sd_scripts_path


# ── Windows Job Object helpers ─────────────────────────────────────────
# When ComfyUI exits, the Job Object is closed and all child processes
# assigned to it are terminated automatically, preventing zombie sd-scripts.

if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectExtendedLimitInformation = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _create_windows_job():
    """Create a Windows Job Object with KILL_ON_JOB_CLOSE flag."""
    if os.name != "nt":
        return None
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    result = _kernel32.SetInformationJobObject(
        job, _JobObjectExtendedLimitInformation,
        ctypes.byref(info), ctypes.sizeof(info),
    )
    if not result:
        _kernel32.CloseHandle(job)
        return None
    return job


def _assign_process_to_job(process, job_handle):
    """Assign a subprocess to a Windows Job Object."""
    if os.name != "nt" or job_handle is None:
        return
    access = _PROCESS_SET_QUOTA | _PROCESS_TERMINATE
    proc_handle = _kernel32.OpenProcess(access, False, process.pid)
    if not proc_handle:
        return
    try:
        _kernel32.AssignProcessToJobObject(job_handle, proc_handle)
    finally:
        _kernel32.CloseHandle(proc_handle)


def _close_job(job_handle):
    """Close a Windows Job Object handle."""
    if os.name != "nt" or job_handle is None:
        return
    _kernel32.CloseHandle(job_handle)


def _terminate_process_tree(process):
    """Terminate a process and its children."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            import signal
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        pass

# Global cache for trained LoRAs
_lora_cache = {}
_cache_file = os.path.join(os.path.dirname(__file__), ".cream_lora_cache.json")


def _load_cache():
    """Load LoRA cache from disk."""
    global _lora_cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                _lora_cache = json.load(f)
        except Exception:
            _lora_cache = {}


def _save_cache():
    """Save LoRA cache to disk."""
    try:
        with open(_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_lora_cache, f, indent=2)
    except Exception:
        pass


def _compute_training_hash(images, captions, ckpt_name, lora_name,
                           target_steps, save_steps, learning_rate,
                           lora_rank, vram_mode, batch_size):
    """Compute a hash of dataset + all training parameters."""
    hasher = hashlib.sha256()

    # Hash file paths and modification times
    for img_path in images:
        hasher.update(img_path.encode('utf-8'))
        if os.path.exists(img_path):
            hasher.update(str(os.path.getmtime(img_path)).encode('utf-8'))

    # Hash all captions
    captions_str = "|".join(captions)

    # Hash all user-configurable settings
    params_str = (f"cream|{captions_str}|{ckpt_name}|{lora_name}"
                  f"|{target_steps}|{save_steps}|{learning_rate}"
                  f"|{lora_rank}|{vram_mode}|{batch_size}|{len(images)}")
    hasher.update(params_str.encode('utf-8'))

    return hasher.hexdigest()[:16]


# Load cache on module import
_load_cache()


class CreamLoraTrainer:
    """
    Trains an SDXL LoRA from a dataset folder using kohya sd-scripts.
    Dataset folder must contain image files with matching .txt caption files.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Get list of checkpoints from ComfyUI
        checkpoints = folder_paths.get_filename_list("checkpoints")

        return {
            "required": {
                "dataset_path": ("STRING", {
                    "default": "",
                    "tooltip": "이미지와 .txt 캡션 파일이 있는 폴더 경로. 각 이미지에 동일 이름의 .txt 캡션 파일이 필요합니다.",
                }),
                "sd_scripts_path": ("STRING", {
                    "default": "",
                    "tooltip": "kohya sd-scripts 설치 경로. sdxl_train_network.py와 venv 폴더가 포함되어 있어야 합니다. 비워두면 자동으로 sd-scripts가 설치됩니다.",
                }),
                "ckpt_name": (checkpoints, {
                    "tooltip": "학습에 사용할 SDXL 베이스 체크포인트.",
                }),
                "lora_name": ("STRING", {
                    "default": "",
                    "tooltip": "저장될 로라 파일 이름. 중간 저장 시 {이름}-step{N}.safetensors 형식으로 저장됩니다.",
                }),
                "target_steps": ("INT", {
                    "default": 1000,
                    "min": 10,
                    "max": 50000,
                    "step": 10,
                    "tooltip": "총 학습 스탭 수. 많을수록 오래 걸리지만 결과가 좋아질 수 있습니다. 소규모 데이터셋 기준 500~2000이 일반적.",
                }),
                "save_steps": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "step": 10,
                    "tooltip": "N 스탭마다 중간 로라를 저장합니다. 0이면 최종 결과만 저장. 학습 단계별 품질 비교에 유용합니다.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0003,
                    "min": 0.00001,
                    "max": 0.1,
                    "step": 0.00001,
                    "tooltip": "U-Net 학습률. Text Encoder LR은 자동으로 이 값의 1/10이 적용됩니다 (High VRAM만). 추천: 0.0001~0.0005.",
                }),
                "lora_rank": ("INT", {
                    "default": 32,
                    "min": 4,
                    "max": 128,
                    "step": 4,
                    "tooltip": "LoRA 차원 수. 높을수록 표현력이 좋지만 파일 크기와 VRAM 사용이 증가합니다. 추천: 16~64.",
                }),
                "vram_mode": (["Low VRAM", "High VRAM"], {
                    "default": "High VRAM",
                    "tooltip": "Low VRAM (8GB+): U-Net만 학습, Text Encoder 출력 캐싱.\nHigh VRAM (12GB+): U-Net과 Text Encoder 모두 학습.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "스탭당 동시 처리 이미지 수. 높을수록 안정적이지만 VRAM 사용 증가. 같은 스탭이라도 배치가 크면 더 많은 이미지를 학습합니다. Low VRAM 모드에서는 1로 고정됩니다.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_dir",)
    OUTPUT_TOOLTIPS = (
        "학습된 LoRA 파일이 저장된 디렉토리 경로.",
    )
    FUNCTION = "train_lora"
    CATEGORY = "training"
    DESCRIPTION = "데이터셋 폴더에서 SDXL LoRA를 학습합니다. (kohya sd-scripts)"

    def train_lora(
        self,
        dataset_path,
        sd_scripts_path,
        ckpt_name,
        lora_name,
        target_steps,
        save_steps,
        learning_rate,
        lora_rank,
        vram_mode,
        batch_size,
    ):
        # ── Validate required STRING inputs ──────────────────────────
        if not dataset_path or not dataset_path.strip():
            raise ValueError("dataset_path를 입력해주세요.")
        if not lora_name or not lora_name.strip():
            raise ValueError("lora_name을 입력해주세요.")

        dataset_path = os.path.expanduser(dataset_path.strip())

        # ── Resolve sd-scripts path (auto-install if empty) ──────
        if not sd_scripts_path or not sd_scripts_path.strip():
            sd_scripts_path = _ensure_sd_scripts()
        else:
            sd_scripts_path = os.path.expanduser(sd_scripts_path.strip())
        lora_name = lora_name.strip()

        # ── Validate dataset path ────────────────────────────────────
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {dataset_path}")

        # Scan for images and their caption files
        images = []
        captions = []
        missing_captions = []

        for filename in sorted(os.listdir(dataset_path)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                img_path = os.path.join(dataset_path, filename)
                base_name = os.path.splitext(filename)[0]
                caption_file = os.path.join(dataset_path, f"{base_name}.txt")

                if not os.path.exists(caption_file):
                    missing_captions.append(filename)
                    continue

                images.append(img_path)
                with open(caption_file, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())

        if missing_captions:
            raise FileNotFoundError(
                f"다음 이미지에 대응하는 .txt 캡션 파일이 없습니다:\n"
                + "\n".join(f"  - {f}" for f in missing_captions)
            )

        if not images:
            raise ValueError(f"데이터셋 경로에 이미지가 없습니다: {dataset_path}")

        num_images = len(images)
        print(f"[Cream LoRA] 이미지 {num_images}장 발견")

        # ── Check cache ──────────────────────────────────────────────
        training_hash = _compute_training_hash(
            images, captions, ckpt_name, lora_name,
            target_steps, save_steps, learning_rate,
            lora_rank, vram_mode, batch_size,
        )

        if training_hash in _lora_cache:
            cached_path = _lora_cache[training_hash]
            if os.path.exists(cached_path):
                print(f"[Cream LoRA] 캐시 적중! 기존 로라 사용: {cached_path}")
                return (cached_path,)
            else:
                # Cached file was deleted, remove stale entry
                del _lora_cache[training_hash]
                _save_cache()

        # ── Validate sd-scripts paths ────────────────────────────────
        train_script = os.path.join(sd_scripts_path, "sdxl_train_network.py")
        if not os.path.exists(train_script):
            raise FileNotFoundError(f"sdxl_train_network.py를 찾을 수 없습니다: {train_script}")

        accelerate_path = _get_accelerate_path(sd_scripts_path)
        if not os.path.exists(accelerate_path):
            raise FileNotFoundError(f"accelerate를 찾을 수 없습니다: {accelerate_path}")

        # ── Validate checkpoint ──────────────────────────────────────
        model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {model_path}")

        # ── Get VRAM preset ──────────────────────────────────────────
        preset = VRAM_PRESETS.get(vram_mode, VRAM_PRESETS["High VRAM"])
        print(f"[Cream LoRA] VRAM 모드: {vram_mode}")
        print(f"[Cream LoRA] 체크포인트: {ckpt_name}")

        # ── Override batch_size for Low VRAM ─────────────────────────
        if vram_mode == "Low VRAM" and batch_size > 1:
            print(f"[Cream LoRA] ⚠️ Low VRAM 모드에서는 batch_size가 1로 고정됩니다. (입력값: {batch_size})")
            batch_size = 1

        # ── Setup output folder ──────────────────────────────────────
        output_folder = os.path.join(dataset_path, "models")
        os.makedirs(output_folder, exist_ok=True)
        lora_output_path = os.path.join(output_folder, f"{lora_name}.safetensors")

        print(f"[Cream LoRA] 저장 경로: {output_folder}")

        # ── Create temp directory with sd-scripts folder structure ───
        # sd-scripts format: train_data_dir/repeats_class/
        temp_dir = tempfile.mkdtemp(prefix="cream_lora_")
        image_folder = os.path.join(temp_dir, "1_subject")  # repeat=1
        os.makedirs(image_folder, exist_ok=True)

        try:
            # Copy images and captions to temp folder
            for idx, (src_path, cap) in enumerate(zip(images, captions)):
                ext = os.path.splitext(src_path)[1]
                dest_path = os.path.join(image_folder, f"image_{idx+1:03d}{ext}")
                shutil.copy2(src_path, dest_path)

                caption_path = os.path.join(image_folder, f"image_{idx+1:03d}.txt")
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(cap)

            print(f"[Cream LoRA] {num_images}장을 임시 폴더에 복사 완료")

            # ── Generate config ──────────────────────────────────────
            text_encoder_lr = learning_rate / 10
            if not preset['network_train_unet_only']:
                print(f"[Cream LoRA] Text Encoder LR: {text_encoder_lr:g} (U-Net LR의 1/10)")
            else:
                print(f"[Cream LoRA] Text Encoder 학습 비활성 (Low VRAM 모드)")

            # Dynamic warmup: 10% of total steps, max 100
            warmup_steps = min(100, max(0, int(target_steps * 0.1)))
            print(f"[Cream LoRA] Warmup: {warmup_steps} steps")

            config_content = generate_training_config(
                name=lora_name,
                image_folder=temp_dir,
                output_folder=output_folder,
                model_path=model_path,
                steps=target_steps,
                save_every_n_steps=save_steps if save_steps > 0 else target_steps,
                learning_rate=learning_rate,
                text_encoder_lr=text_encoder_lr,
                lora_rank=lora_rank,
                lora_alpha=lora_rank,  # alpha = rank for full strength
                batch_size=batch_size,
                lr_warmup_steps=warmup_steps,
                network_train_unet_only=preset['network_train_unet_only'],
                cache_text_encoder_outputs=preset['cache_text_encoder_outputs'],
            )

            config_path = os.path.join(temp_dir, "training_config.toml")
            save_config(config_content, config_path)
            print(f"[Cream LoRA] 학습 설정 저장: {config_path}")

            # ── Build and run training command ───────────────────────
            cmd = [
                accelerate_path,
                "launch",
                "--num_cpu_threads_per_process=2",
            ]

            # Pass mixed_precision from config to accelerate
            config_text = open(config_path, 'r', encoding='utf-8').read()
            mp_match = MIXED_PRECISION_PATTERN.search(config_text)
            if mp_match:
                mp_value = mp_match.group(1).strip().lower()
                if mp_value in ("fp16", "bf16", "fp8", "no"):
                    cmd.extend(["--mixed_precision", mp_value])

            cmd.extend([
                train_script,
                f"--config_file={config_path}",
            ])

            # ── Release VRAM before training ─────────────────────────
            if comfy_model_management is not None:
                print("[Cream LoRA] ComfyUI 모델 언로드 중 (VRAM 확보)...")
                comfy_model_management.unload_all_models()
                soft_empty = getattr(comfy_model_management, "soft_empty_cache", None)
                if callable(soft_empty):
                    soft_empty()

            print(f"[Cream LoRA] 학습 시작: {lora_name}")
            print(f"[Cream LoRA] 이미지: {num_images}, 스탭: {target_steps}, LR: {learning_rate}, Rank: {lora_rank}")

            # Run training subprocess with process management
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env.setdefault('PYTHONUTF8', '1')

            creationflags = 0
            popen_kwargs = {}
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            job_handle = _create_windows_job() if os.name == "nt" else None

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=sd_scripts_path,
                env=env,
                bufsize=1,
                creationflags=creationflags,
                **popen_kwargs,
            )
            _assign_process_to_job(process, job_handle)

            # Stream output with interrupt support (thread + queue)
            try:
                output_queue = queue.Queue()
                output_lines = []

                def _reader():
                    try:
                        for line in process.stdout:
                            output_queue.put(line)
                    finally:
                        output_queue.put(None)

                reader_thread = threading.Thread(target=_reader, daemon=True)
                reader_thread.start()

                while True:
                    try:
                        line = output_queue.get(timeout=0.3)
                    except queue.Empty:
                        line = None

                    # Check for ComfyUI interrupt (Cancel button)
                    if comfy_model_management is not None:
                        try:
                            comfy_model_management.throw_exception_if_processing_interrupted()
                        except Exception:
                            print("[Cream LoRA] ComfyUI 인터럽트 감지 — 학습 중단 중...")
                            _terminate_process_tree(process)
                            raise

                    if line is None:
                        if process.poll() is not None and not reader_thread.is_alive() and output_queue.empty():
                            break
                        continue

                    text = line.rstrip()
                    if text:
                        output_lines.append(text)
                        print(f"[sd-scripts] {text}")

                process.wait()

                if process.returncode != 0:
                    # Save log file for debugging
                    from datetime import datetime
                    log_name = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                    log_path = os.path.join(output_folder, log_name)
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write("\n".join(output_lines))
                    print(f"[Cream LoRA] 오류 로그 저장: {log_path}")
                    raise RuntimeError(f"sd-scripts 학습 실패 (코드: {process.returncode})")
            except BaseException:
                _terminate_process_tree(process)
                raise
            finally:
                _close_job(job_handle)

            print(f"[Cream LoRA] 학습 완료!")

            # ── Find the trained LoRA ────────────────────────────────
            if not os.path.exists(lora_output_path):
                # Check for alternative naming
                possible_files = [
                    f for f in os.listdir(output_folder)
                    if f.startswith(lora_name) and f.endswith('.safetensors')
                ]
                if possible_files:
                    lora_output_path = os.path.join(output_folder, sorted(possible_files)[-1])
                else:
                    raise FileNotFoundError(f"학습된 LoRA 파일을 찾을 수 없습니다: {output_folder}")

            # ── Remove duplicate step file at final step ─────────────
            if save_steps > 0 and target_steps % save_steps == 0:
                dup_name = f"{lora_name}-step{target_steps:08d}.safetensors"
                dup_path = os.path.join(output_folder, dup_name)
                if os.path.exists(dup_path):
                    os.remove(dup_path)
                    print(f"[Cream LoRA] 중복 파일 삭제: {dup_name}")

            # ── Save to cache ────────────────────────────────────────
            _lora_cache[training_hash] = output_folder
            _save_cache()

            print(f"[Cream LoRA] LoRA 저장 완료: {output_folder}")

            return (output_folder,)

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[Cream LoRA] 임시 폴더 정리 실패: {e}")
