import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiofiles
import aiofiles.os
import aiohttp
import torch
from loguru import logger

# === Model config ===
MODEL_DOWNLOAD_URL = "https://YOUR_PUBLIC_MODEL_URL/kokoro-v1_0.pth"
MODEL_FILENAME = "v1_0/kokoro-v1_0.pth"
MODEL_LOCAL_DIR = "/tmp/kokoro"

# === General utilities ===

async def _find_file(
    filename: str,
    search_paths: List[str],
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    if os.path.isabs(filename) and await aiofiles.os.path.exists(filename):
        return filename

    for path in search_paths:
        full_path = os.path.join(path, filename)
        if await aiofiles.os.path.exists(full_path):
            if filter_fn is None or filter_fn(full_path):
                return full_path

    raise FileNotFoundError(f"File not found: {filename} in paths: {search_paths}")


# === Main fix: model path and download ===

async def get_model_path(model_name: str) -> str:
    """Ensure the model is available locally and return the file path."""
    full_path = os.path.join(MODEL_LOCAL_DIR, model_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Download if missing
    if not os.path.exists(full_path):
        logger.info(f"Downloading model to {full_path}...")
        async with aiohttp.ClientSession() as session:
            async with session.get(MODEL_DOWNLOAD_URL) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to download model: {resp.status}")
                content = await resp.read()
                with open(full_path, "wb") as f:
                    f.write(content)

    return full_path


# === Optional: helper to check model exists ===

async def verify_model_path(model_path: str) -> bool:
    return await aiofiles.os.path.exists(model_path)

