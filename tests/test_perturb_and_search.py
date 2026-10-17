from PIL import Image

from agentverify import perturb
from agentverify.tools.search import EvidenceTool, is_trusted, rerank_by_trust


def test_image_perturbations_keep_valid_images():
    im = Image.new("RGB", (64, 64), (120, 40, 200))
    for name, fn in perturb.IMAGE_PERTURBATIONS.items():
        for sev in (1, 2, 3):
            out = fn(im, sev)
            assert out.size[0] > 0 and out.mode == "RGB", name
    assert perturb.center_crop(im, 3).size[0] < perturb.center_crop(im, 1).size[0]


def test_text_perturbations_deterministic_and_changing():
    t = "The mayor opened the new bridge on Tuesday"
    assert perturb.typos(t, 3, seed=1) == perturb.typos(t, 3, seed=1)
    assert perturb.typos(t, 3, seed=1) != t
    assert perturb.shout(t, 2).endswith("!!")
    assert perturb.hashtags(t, 2).startswith(t)


def test_trust_and_cache(tmp_path):
    assert is_trusted("https://www.reuters.com/x") and is_trusted("https://en.wikipedia.org/wiki/A")
    assert not is_trusted("https://randomblog.example/x")
    calls = []

    def fake_backend(q, k):
        calls.append(q)
        return [{"title": "blog", "url": "https://randomblog.example/a", "snippet": "s"},
                {"title": "Reuters", "url": "https://www.reuters.com/a", "snippet": "s"}]

    tool = EvidenceTool(backends={"web": fake_backend}, cache_dir=tmp_path)
    items, c1 = tool.search("bridge opening", "web")
    items2, c2 = tool.search("bridge opening", "web")
    assert c1.search_calls == 1 and c2.search_calls == 0 and len(calls) == 1   # second hit is cached
    assert items[0].trusted and items[0].domain == "reuters.com"
    assert [e.title for e in items2] == [e.title for e in items]
