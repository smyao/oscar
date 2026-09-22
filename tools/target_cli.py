# 档案 #28/#29/#30/#84：建引擎前初始化父进程线程池；保留附录 A 参数及 MTP eager 作用域。
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def serve_argv(config: dict) -> list[str]:
    args = ["serve", config["model"]]
    for key in ("served_model_name", "host", "port", "data_parallel_size", "tensor_parallel_size",
                "max_model_len", "max_num_batched_tokens", "max_num_seqs", "gpu_memory_utilization", "quantization"):
        args += ["--" + key.replace("_", "-"), str(config[key])]
    for key in ("compilation_config", "speculative_config", "additional_config", "hf_overrides"):
        # Keep the user-provided spelling for speculative_config.
        flag = "--speculative_config" if key == "speculative_config" else "--" + key.replace("_", "-")
        args += [flag, json.dumps(config[key], separators=(",", ":"))]
    args += ["--trust-remote-code", "--async-scheduling", "--allowed-local-media-path", "/",
             "--mm-processor-cache-gb", "0", "--mamba-cache-dtype", "bfloat16", "--mamba-ssm-cache-dtype", "bfloat16"]
    return args


def target_env(config: dict, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    devices = config.get("devices")
    if not isinstance(devices, list) or len(devices) != 4 or len(set(devices)) != 4 or any(type(x) is not int or x < 0 for x in devices):
        raise ValueError("configs/target.json devices must contain the four physical NPU IDs selected for this task")
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    env["OSCAR_ENABLED"] = "1"
    plugins = [p.strip() for p in env.get("VLLM_PLUGINS", "").split(",") if p.strip()]
    # With no allowlist vLLM loads every registered plugin. Do not accidentally exclude Ascend.
    if plugins:
        for required in ("ascend", "oscar_ascend"):
            if required not in plugins:
                plugins.append(required)
        env["VLLM_PLUGINS"] = ",".join(plugins)
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    command = serve_argv(config)
    if args.print_command:
        import shlex
        print("vllm " + shlex.join(command))
        return 0
    os.environ.update(target_env(config))
    from oscar_ascend.integration.runtime_api import require_runtime
    require_runtime().assert_ready()
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    # Same order as the engine, archive #28/#78.
    import vllm_ascend.ops
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", *command]
    vllm_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
