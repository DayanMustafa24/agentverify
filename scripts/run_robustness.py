"""Gap 2: robustness. Runs B2 and the full agent system on clean vs perturbed copies of D_rob posts.
  python scripts/run_robustness.py --root data/raw/MMFakeBench --n 60 --shard 0/3
Calls per post = 3 conditions x (5 full + 1 b2) = 18 (crop is real-only and off by default). Rerun after a quota stop: cache resumes.
--save-examples K copies K perturbed images + captions per condition to data/cache/perturb_check/ for the HAND CHECK."""
import argparse
import random
import shutil
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _common import get_posts  # noqa: E402

from agentverify.agents.common import QuotaGuard, RateLimited, cached_call  # noqa: E402
from agentverify.agents.orchestrator import run_pipeline  # noqa: E402
from agentverify.baselines.single_vlm import parse_reply  # noqa: E402
from agentverify.config import LLMConfig  # noqa: E402
from agentverify.contracts import RunRecord  # noqa: E402
from agentverify.eval.report import write_records  # noqa: E402
from agentverify.llm import OpenAICompatClient  # noqa: E402
from agentverify.prompts import B2_SYSTEM, B2_USER  # noqa: E402
from agentverify.robustness import make_condition_post  # noqa: E402
from agentverify.tools.search import EvidenceTool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--n", type=int, default=60, help="total stratified D_rob sample (all shards)")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--conditions", nargs="+", default=["clean", "jpeg3", "typos3"])
ap.add_argument("--configs", nargs="+", default=["b2", "full"], choices=["b2", "full"])
ap.add_argument("--rpm", type=float, default=12)
ap.add_argument("--max-calls", type=int, default=450)
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--save-examples", type=int, default=0)
ap.add_argument("--out-dir", default="results/robust")
args = ap.parse_args()

posts = sorted(get_posts(args.root, "rob"), key=lambda p: p.post_id)
rng = random.Random(42)
groups = defaultdict(list)
for p in posts:
    groups[p.gold_source.value].append(p)
picked = []
for g in groups.values():
    picked += rng.sample(g, min(max(1, round(args.n * len(g) / len(posts))), len(g)))
picked.sort(key=lambda p: p.post_id)
i, k = map(int, args.shard.split("/"))
posts = picked[i::k]
print(f"D_rob total={len(picked)} this shard={len(posts)} conditions={args.conditions}")

cfg = LLMConfig.from_env()
cache_dir = f"data/cache/agents/{cfg.model.replace('/', '_')}"
client = RateLimited(OpenAICompatClient(cfg), rpm=args.rpm, max_calls=args.max_calls)
tool = EvidenceTool()
pert_dir = "data/cache/perturbed"
stop = threading.Event()
saved = defaultdict(int)
lock = threading.Lock()


def work(post):
    out = {}
    for cond in args.conditions:
        pp = make_condition_post(post, cond, pert_dir)
        if pp is None:
            continue
        if args.save_examples and cond != "clean":
            with lock:
                if saved[cond] < args.save_examples:
                    saved[cond] += 1
                    d = Path("data/cache/perturb_check") / cond
                    d.mkdir(parents=True, exist_ok=True)
                    tag = f"{saved[cond]:02d}_{post.gold_source.value}"
                    shutil.copy(pp.image_path, d / f"{tag}.png")
                    (d / f"{tag}.txt").write_text(f"ORIGINAL: {post.text}\nPERTURBED: {pp.text}\nGOLD: {post.gold_source.value}\n",
                                                  encoding="utf-8")
        for c in args.configs:
            if stop.is_set():
                return out
            if c == "b2":
                reply, cost = cached_call(cache_dir, "b2", pp.post_id, client, B2_SYSTEM,
                                          B2_USER.format(text=pp.text), pp.image_path)
                out[(cond, c)] = RunRecord(method="b2_single_vlm", post=pp,
                                           final=parse_reply(pp.post_id, reply), cost=cost)
            else:
                out[(cond, c)] = run_pipeline(pp, client, tool, cache_dir, method="agent_full",
                                              trace_path=f"logs/traces/rob_{cond}_full_shard{i}.jsonl")
    return out


results = defaultdict(list)
with ThreadPoolExecutor(args.workers) as ex:
    futs = [ex.submit(work, p) for p in posts]
    for f in tqdm(as_completed(futs), total=len(futs)):
        try:
            for key, rec in f.result().items():
                results[key].append(rec)
        except QuotaGuard as e:
            stop.set()
            print(f"[quota guard] {e}. Rerun later; cached calls are skipped.")
        except Exception as e:
            print(f"[post failed] {type(e).__name__}: {e}")

out_dir = Path(args.out_dir) / f"shard{i}"
for (cond, c), recs in results.items():
    write_records(str(out_dir / f"rob_{cond}_{c}.jsonl"), recs)
    print(f"{cond}/{c}: {len(recs)} records")
print(f"API calls used this run: {client.calls}")
