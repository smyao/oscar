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
    if config.get("profiler_config") is not None:
        args += ["--profiler-config",json.dumps(config["profiler_config"],separators=(",",":"))]
    return args


def profiler_config_env(config: dict, env: dict[str, str] | None = None) -> dict | None:
    """Arm the native profiler via OSCAR_PROFILE_DIR without editing target.json.

    An explicit profiler_config in the target config always wins. The native
    wrapper stays idle until the probe's /start_profile window; serving and
    numerics are unchanged outside the bounded measurement.
    """
    configured = config.get("profiler_config")
    if configured is not None:
        return configured
    directory = str((os.environ if env is None else env).get("OSCAR_PROFILE_DIR", "")).strip()
    if not directory:
        return None
    parts = Path(directory).parts
    if ".." in parts or directory.endswith("/"):
        raise ValueError("OSCAR_PROFILE_DIR must be a plain directory path")
    return {"profiler": "torch", "torch_profiler_dir": str(Path(directory).resolve())}


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
    parser.add_argument("--native",action="store_true",help="explicit unmodified native baseline, never selected after an OSCAR failure")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    profiler = profiler_config_env(config)
    if profiler is not config.get("profiler_config"):
        config = dict(config, profiler_config=profiler)
    command = serve_argv(config)
    if args.print_command:
        import shlex
        print("vllm " + shlex.join(command))
        return 0
    os.environ.update(target_env(config))
    os.environ["OSCAR_TARGET_CONFIG"]=str(args.config.resolve())
    if args.native:
        os.environ["OSCAR_ENABLED"]="0"
    else:
        from oscar_ascend.plugin import register
        register()
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
