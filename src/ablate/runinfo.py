import hashlib, json, subprocess, os
from omegaconf import OmegaConf

def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"

def git_dirty():
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        return None

def config_hash(cfg):
    s = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:8]

def run_name(cfg):
    return f"{cfg.exp}-{git_hash()}-{config_hash(cfg)}-s{cfg.seed}"

def metadata(cfg):
    return {
        "git": git_hash(), "dirty": git_dirty(),
        "config_hash": config_hash(cfg), "seed": cfg.seed,
        "image_id": os.environ.get("IMAGE_ID", "unknown"),
    }
