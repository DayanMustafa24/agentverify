from PIL import Image

from agentverify.contracts import PostInput, SourceLabel
from agentverify.robustness import make_condition_post, original_id


def _post(tmp_path, src):
    p = tmp_path / "a.png"
    Image.new("RGB", (64, 64), (120, 30, 200)).save(p)
    return PostInput(post_id="abc", text="The mayor opened a new bridge today", image_path=str(p),
                     gold_is_fake=src != SourceLabel.REAL, gold_source=src)


def test_perturbed_copy_gets_new_id_and_keeps_label(tmp_path):
    post = _post(tmp_path, SourceLabel.IMAGE_FALSE)
    q = make_condition_post(post, "jpeg3", tmp_path / "out")
    assert q.post_id == "abc__jpeg3" and original_id(q.post_id) == "abc"
    assert q.image_path != post.image_path
    assert q.gold_source == post.gold_source and q.gold_is_fake == post.gold_is_fake


def test_crop_is_real_only(tmp_path):
    assert make_condition_post(_post(tmp_path, SourceLabel.MISMATCH), "crop2", tmp_path / "o") is None
    assert make_condition_post(_post(tmp_path, SourceLabel.REAL), "crop2", tmp_path / "o") is not None


def test_text_perturbation_is_deterministic(tmp_path):
    post = _post(tmp_path, SourceLabel.TEXT_FALSE)
    a = make_condition_post(post, "typos3", tmp_path / "o")
    b = make_condition_post(post, "typos3", tmp_path / "o")
    assert a.text == b.text and a.image_path == post.image_path
