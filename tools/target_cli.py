# 档案 #28/#29/#30/#84：建引擎前初始化父进程线程池；保留附录 A 参数及 MTP eager 作用域。
# #34/#36：原生预算模式不传预算参数；worker按原生有效容量预分配。
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from .model_config import add_model_arguments, apply_model_options

ROOT = Path(__file__).resolve().parents[1]


def diagnose_guard_enabled(config: dict) -> bool:
    """Explicit #157-159 fault attribution; never a performance setting."""
    enabled = config.get("diagnose_guard", False)
    if type(enabled) is not bool:
        raise ValueError("diagnose_guard must be an explicit boolean")
    return enabled


def serve_argv(config: dict) -> list[str]:
    config = apply_model_options(config)
    native_budget = config.get("use_native_token_budget", True)
    if type(native_budget) is not bool:
        raise ValueError("use_native_token_budget must be an explicit boolean")
    # User 2026-10-13 stable branch: explicit budget is intentional; no
    # scheduler substitution or prefix-cache override is inferred from it.
    if (config.get("scheduler_cls") is not None or
            config.get("experimental_whole_prefill", False) or
            config.get("experimental_bounded_whole_prefill", False)):
        raise ValueError("token budget requires the native scheduler")
    budget = config.get("max_num_batched_tokens")
    if native_budget and "max_num_batched_tokens" in config:
        raise ValueError("native token budget requires an omitted budget")
    if not native_budget and (type(budget) is not int or budget <= 0):
        raise ValueError("explicit token budget must be a positive integer")
    args = ["serve", config["model"]]
    if not native_budget:
        args += ["--max-num-batched-tokens", str(budget)]
    for key in ("served_model_name", "host", "port", "data_parallel_size", "tensor_parallel_size",
                "max_model_len", "max_num_seqs", "gpu_memory_utilization", "dtype"):
        args += ["--" + key.replace("_", "-"), str(config[key])]
    if config["quantization"] == "ascend":
        args += ["--quantization", "ascend"]
    for key in ("compilation_config", "speculative_config", "additional_config", "hf_overrides"):
        # Keep the user-provided spelling for speculative_config.
        flag = "--speculative_config" if key == "speculative_config" else "--" + key.replace("_", "-")
        args += [flag, json.dumps(config[key], separators=(",", ":"))]
    args += ["--trust-remote-code", "--async-scheduling", "--allowed-local-media-path", "/",
             "--mm-processor-cache-gb", "0", "--mamba-cache-dtype", "bfloat16", "--mamba-ssm-cache-dtype", "bfloat16"]
    # Archive #26/#37-49/#127: prefix sharing changes native GDN geometry.
    # User 2026-10-13 stable branch policy: disable APC by default. This is
    # an explicit launch policy, never an automatic response to a failure.
    enabled = config.get("enable_prefix_caching", False)
    if type(enabled) is not bool:
        raise ValueError("enable_prefix_caching must be an explicit boolean")
    args.append("--enable-prefix-caching" if enabled else "--no-enable-prefix-caching")
    if "scheduler_cls" in config:
        scheduler = config["scheduler_cls"]
        if (not isinstance(scheduler, str) or "." not in scheduler
                or not all(part.isidentifier() for part in scheduler.split("."))):
            raise ValueError("scheduler_cls must be an explicit Python class path")
        args += ["--scheduler-cls", scheduler]
    if config.get("profiler_config") is not None:
        args += ["--profiler-config",json.dumps(config["profiler_config"],separators=(",",":"))]
    if diagnose_guard_enabled(config):
        # Main-model graph guards are otherwise fused at capture and can
        # only surface as prior_work in a later eager diagnostic guard.
        args.append("--enforce-eager")
    return args


def target_env(config: dict, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if config.get("experimental_bf16_attention",False) and not diagnose_guard_enabled(config):
        # User paused performance work. A stale inherited observer control
        # must not install NPU Event sampling in a normal BF16 service.
        env.pop("OSCAR_PASSIVE_TIMING_CONTROL",None)
    if diagnose_guard_enabled(config):
        env["OSCAR_GUARD_HOST_SYNC"] = "1"
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/target.json")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--native",action="store_true",help="explicit unmodified native baseline, never selected after an OSCAR failure")
    add_model_arguments(parser)
    args = parser.parse_args(argv)
    original = json.loads(args.config.read_text())
    config = apply_model_options(original, args, require_model=not args.print_command)
    command = serve_argv(config)
    if args.print_command:
        import shlex
        print("vllm " + shlex.join(command))
        return 0
    if config != original:
        # Workers consume the same model choice as the native CLI, including
        # direct target_cli overrides. Keep it alive until vLLM exits.
        with tempfile.TemporaryDirectory(prefix="oscar-target-") as directory:
            effective = Path(directory) / "effective-target.json"
            effective.write_text(json.dumps(config, indent=2) + "\n")
            return _run_service(config, effective, command, native=args.native)
    return _run_service(config, args.config, command, native=args.native)


def _run_service(config: dict, config_path: Path, command: list[str], *, native: bool) -> int:
    env = target_env(config)
    if "OSCAR_PASSIVE_TIMING_CONTROL" not in env:
        os.environ.pop("OSCAR_PASSIVE_TIMING_CONTROL",None)
    os.environ.update(env)
    os.environ["OSCAR_TARGET_CONFIG"]=str(config_path.resolve())
    if diagnose_guard_enabled(config):
        print("[oscar] PERF_DIAGNOSTIC mode=trap target_eager=true "
              "guard_host_sync=1 requires_user_load=true performance_valid=false "
              "bug_fixed=not_established", flush=True)
    if native:
        os.environ["OSCAR_ENABLED"]="0"
        if os.environ.get("OSCAR_PASSIVE_TIMING_CONTROL") or os.environ.get("OSCAR_TIMER_CONTROL"):
            # Passive runner/graph timing only; platform/cache routing stays
            # native and no OSCAR runtime is installed.
            from oscar_ascend.plugin import register_passive_observer
            register_passive_observer()
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
