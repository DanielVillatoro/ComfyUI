"""Queue a FLUX GGUF API workflow through ComfyUI and save results next to this script.

Usage: python3 run_flux_gguf.py [workflow.json] [-n COUNT] [--seed SEED]
The prompt lives in the workflow file (node "4"); seeds are random unless --seed is given.
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

SERVER = "http://127.0.0.1:8188"
HERE = Path(__file__).resolve().parent


def api(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(SERVER + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def run_one(workflow, seed):
    wf = json.loads(workflow.read_text())
    prefix = wf["10"]["inputs"]["filename_prefix"]
    # KSampler takes "seed"; RandomNoise (custom sampling) takes "noise_seed".
    for node in wf.values():
        for key in ("seed", "noise_seed"):
            if key in node["inputs"]:
                node["inputs"][key] = seed
    wf["10"]["inputs"]["filename_prefix"] = f"{prefix}_{seed}"

    t0 = time.time()
    prompt_id = json.loads(api("/prompt", {"prompt": wf}))["prompt_id"]
    while True:
        hist = json.loads(api(f"/history/{prompt_id}")).get(prompt_id)
        if hist and (hist["status"].get("completed") or hist["status"].get("status_str") == "error"):
            break
        time.sleep(2)
    wall = time.time() - t0

    if hist["status"].get("status_str") != "success":
        raise RuntimeError(json.dumps(hist["status"], indent=2))

    # Render time from ComfyUI's own execution_start -> execution_success timestamps (ms).
    stamps = {name: msg["timestamp"] for name, msg in hist["status"]["messages"] if "timestamp" in msg}
    render = (stamps["execution_success"] - stamps["execution_start"]) / 1000

    img = hist["outputs"]["10"]["images"][0]
    query = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
    dest = HERE / f"{prefix}_seed{seed}.png"
    dest.write_bytes(api(f"/view?{query}"))
    return render, wall, dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", nargs="?", default=str(HERE / "flux_gguf_workflow_api.json"))
    parser.add_argument("-n", "--count", type=int, default=1)
    parser.add_argument("--seed", type=int, help="fixed seed for the first image; the rest stay random")
    args = parser.parse_args()

    seeds = [random.randint(0, 2**32 - 1) for _ in range(args.count)]
    if args.seed is not None:
        seeds[0] = args.seed
    for i, seed in enumerate(seeds, 1):
        render, wall, dest = run_one(Path(args.workflow), seed)
        print(f"[{i}/{args.count}] seed={seed} render={render:.1f}s wall={wall:.1f}s file={dest}", flush=True)


if __name__ == "__main__":
    main()
