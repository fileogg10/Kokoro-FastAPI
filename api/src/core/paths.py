"""Async file and path operations."""

import io
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiofiles
import aiofiles.os
import torch
from loguru import logger

from .config import settings


async def _find_file(
    filename: str,
    search_paths: List[str],
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> str:
    """Find file in search paths.

    Args:
        filename: Name of file to find
        search_paths: List of paths to search in
        filter_fn: Optional function to filter files

    Returns:
        Absolute path to file

    Raises:
        RuntimeError: If file not found
    """
    if os.path.isabs(filename) and await aiofiles.os.path.exists(filename):
        return filename

    for path in search_paths:
        full_path = os.path.join(path, filename)
        if await aiofiles.os.path.exists(full_path):
            if filter_fn is None or filter_fn(full_path):
                return full_path

    raise FileNotFoundError(f"File not found: {filename} in paths: {search_paths}")


async def _scan_directories(
    search_paths: List[str], filter_fn: Optional[Callable[[str], bool]] = None
) -> Set[str]:
    """Scan directories for files.

    Args:
        search_paths: List of paths to scan
        filter_fn: Optional function to filter files

    Returns:
        Set of matching filenames
    """
    results = set()

    for path in search_paths:
        if not await aiofiles.os.path.exists(path):
            continue

        try:
            entries = await aiofiles.os.scandir(path)
            for entry in entries:
                if filter_fn is None or filter_fn(entry.name):
                    results.add(entry.name)
        except Exception as e:
            logger.warning(f"Error scanning {path}: {e}")

    return results


async def get_model_path(model_name: str) -> str:
    """Get path to model file.

    Args:
        model_name: Name of model file

    Returns:
        Absolute path to model file

    Raises:
        RuntimeError: If model not found
    """
    model_dir = os.environ.get("KOKORO_MODEL_DIR", os.path.join(os.path.dirname(__file__), "../../models"))
    os.makedirs(model_dir, exist_ok=True)
    logger.debug(f"Searching for model in path: {model_dir}")
    return await _find_file(model_name, [model_dir])


async def get_voice_path(voice_name: str) -> str:
    """Get path to voice file.

    Args:
        voice_name: Name of voice file (without .pt extension)

    Returns:
        Absolute path to voice file

    Raises:
        RuntimeError: If voice not found
    """
    voice_dir = os.environ.get("KOKORO_MODEL_DIR", "/tmp/kokoro_models")
    os.makedirs(voice_dir, exist_ok=True)
    voice_file = f"{voice_name}.pt"
    logger.debug(f"Searching for voice in path: {voice_dir}")
    return await _find_file(voice_file, [voice_dir])


async def list_voices() -> List[str]:
    """List available voice files.

    Returns:
        List of voice names (without .pt extension)
    """
    voice_dir = os.environ.get("KOKORO_MODEL_DIR", "/tmp/kokoro_models")
    os.makedirs(voice_dir, exist_ok=True)
    logger.debug(f"Scanning for voices in path: {voice_dir}")

    def filter_voice_files(name: str) -> bool:
        return name.endswith(".pt")

    voices = await _scan_directories([voice_dir], filter_voice_files)
    return sorted([name[:-3] for name in voices])


async def load_voice_tensor(
    voice_path: str, device: str = "cpu", weights_only=False
) -> torch.Tensor:
    try:
        async with aiofiles.open(voice_path, "rb") as f:
            data = await f.read()
            return torch.load(io.BytesIO(data), map_location=device, weights_only=weights_only)
    except Exception as e:
        raise RuntimeError(f"Failed to load voice tensor from {voice_path}: {e}")


async def save_voice_tensor(tensor: torch.Tensor, voice_path: str) -> None:
    try:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        async with aiofiles.open(voice_path, "wb") as f:
            await f.write(buffer.getvalue())
    except Exception as e:
        raise RuntimeError(f"Failed to save voice tensor to {voice_path}: {e}")


async def load_json(path: str) -> dict:
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON file {path}: {e}")


async def load_model_weights(path: str, device: str = "cpu") -> dict:
    try:
        async with aiofiles.open(path, "rb") as f:
            data = await f.read()
            return torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {path}: {e}")


async def read_file(path: str) -> str:
    try:
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file {path}: {e}")


async def read_bytes(path: str) -> bytes:
    try:
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read file {path}: {e}")


async def get_web_file_path(filename: str) -> str:
    web_dir = os.path.join("/app", settings.web_player_path)
    logger.debug(f"Searching for web file in path: {web_dir}")
    return await _find_file(filename, [web_dir])


async def get_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".html": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
    }.get(ext, "application/octet-stream")


async def verify_model_path(model_path: str) -> bool:
    return await aiofiles.os.path.exists(model_path)


async def cleanup_temp_files() -> None:
    try:
        if not await aiofiles.os.path.exists(settings.temp_file_dir):
            await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
            return

        entries = await aiofiles.os.scandir(settings.temp_file_dir)
        for entry in entries:
            if entry.is_file():
                stat = await aiofiles.os.stat(entry.path)
                max_age = stat.st_mtime + (settings.max_temp_dir_age_hours * 3600)
                if max_age < stat.st_mtime:
                    try:
                        await aiofiles.os.remove(entry.path)
                        logger.info(f"Cleaned up old temp file: {entry.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old temp file {entry.name}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning temp files: {e}")


async def get_temp_file_path(filename: str) -> str:
    temp_path = os.path.join(settings.temp_file_dir, filename)
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        await aiofiles.os.makedirs(settings.temp_file_dir, exist_ok=True)
    return temp_path


async def list_temp_files() -> List[str]:
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        return []

    entries = await aiofiles.os.scandir(settings.temp_file_dir)
    return [entry.name for entry in entries if entry.is_file()]


async def get_temp_dir_size() -> int:
    if not await aiofiles.os.path.exists(settings.temp_file_dir):
        return 0

    total = 0
    entries = await aiofiles.os.scandir(settings.temp_file_dir)
    for entry in entries:
        if entry.is_file():
            stat = await aiofiles.os.stat(entry.path)
            total += stat.st_size
    return total
