"""Run the agent pipeline (and ablations / B2) with caching, rate limiting and sharding.
  python scripts/run_agent.py --root data/raw/MMFakeBench --role eval --n 150 --shard 0/3 --configs b2 full no_text no_image no_evidence
  python scripts/run_agent.py --root data/raw/MMFakeBench --role cal  --n 60  --shard 0/3 --configs full
Rerun the same command after a quota stop: finished calls are cached and skipped."""
import argparse
import random
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
from agentverify.tools.search import EvidenceTool  # noqa: E402

CONFIGS = {"full": {}, "no_text": {"use_text": False}, "no_image": {"use_image": False},
           "no_evidence": {"use_evidence": False}}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--role", choices=["eval", "cal"], required=True)
ap.add_argument("--n", type=int, required=True, help="total stratified sample size (all shards together)")
ap.add_argument("--shard", default="0/1", help="i/k: this member runs shard i of k")
ap.add_argument("--configs", nargs="+", default=["full"], choices=["b2"] + list(CONFIGS))
ap.add_argument("--rpm", type=float, default=12)
ap.add_argument("--max-calls", type=int, default=450, help="stop before the daily quota (RPD)")
ap.add_argument("--workers", type=int, default=5)
ap.add_argument("--out-dir", default="results/agent")
args = ap.parse_args()

split = "val" if args.role == "eval" else "test"   # calibration posts never overlap the evaluation posts
posts = sorted(get_posts(args.root, split), key=lambda p: p.post_id)
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
print(f"role={args.role} split={split} total={len(picked)} this shard={len(posts)}")

cfg = LLMConfig.from_env()
cache_dir = f"data/cache/agents/{cfg.model.replace('/', '_')}"
client = RateLimited(OpenAICompatClient(cfg), rpm=args.rpm, max_calls=args.max_calls)
tool = EvidenceTool()
out_dir = Path(args.out_dir) / f"shard{i}"
trace = lambda c: f"logs/traces/{args.role}_{c}_shard{i}.jsonl"  # noqa: E731
stop = threading.Event()


def work(post):
    out = {}
    for c in args.configs:
        if stop.is_set():
            break
        if c == "b2":
            reply, cost = cached_call(cache_dir, "b2", post.post_id, client, B2_SYSTEM,
                                      B2_USER.format(text=post.text), post.image_path)
            out[c] = RunRecord(method="b2_single_vlm", post=post, final=parse_reply(post.post_id, reply), cost=cost)
        else:
            out[c] = run_pipeline(post, client, tool, cache_dir, method=f"agent_{c}",
                                  trace_path=trace(c), **CONFIGS[c])
    return out


results = defaultdict(list)
with ThreadPoolExecutor(args.workers) as ex:
    futs = [ex.submit(work, p) for p in posts]
    for f in tqdm(as_completed(futs), total=len(futs)):
        try:
            for c, rec in f.result().items():
                results[c].append(rec)
        except QuotaGuard as e:
            stop.set()
            print(f"[quota guard] {e}. Rerun this command later; cached calls are skipped.")
        except Exception as e:
            print(f"[post failed] {type(e).__name__}: {e}")

for c, recs in results.items():
    write_records(str(out_dir / f"{args.role}_{c}.jsonl"), recs)
    print(f"{c}: wrote {len(recs)} records (a post counts only if all its configs finished)")
print(f"API calls used this run: {client.calls}")
