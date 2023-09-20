from pathlib import Path


def get_asset_path(file_name: str) -> Path:
    return Path(__file__).parent / "assets" / file_name
