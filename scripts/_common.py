from pathlib import Path

from agentverify.data.mmfakebench import load_split
from agentverify.data.splits import load_ids


def get_posts(root, split_name, splits_dir="data/splits", limit=None):
    """split_name: val | train | cal | test | rob  (val is official; others come from the test file)."""
    base = "val" if split_name == "val" else "test"
    posts = load_split(root, base)
    if split_name != "val" or (Path(splits_dir) / "val.json").exists():
        ids = set(load_ids(splits_dir, split_name))
        posts = [p for p in posts if p.post_id in ids]
    return posts[:limit] if limit else posts
