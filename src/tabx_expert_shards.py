"""Published expert shards have exactly a ten-digit seed filename."""
from pathlib import Path
import re


def published_expert_shards(directory):
    return sorted(path for path in Path(directory).iterdir()
                  if re.fullmatch(r"[0-9]{10}\.h5", path.name) and path.is_file())
