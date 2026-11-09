"""Archive #34/#36/#127/#128/#157/#172/#174: narrow authorized host extensions.

Reverse ONLY the exact reviewed offload/probe seams before comparing the
remaining file against its existing accepted hash. CANN files are never
normalized, and no wildcard AST/file/hash exemption is permitted here.
"""
import json

from test_stable_launch_frozen import AUTHORIZED_KV_OFFLOAD


_RUNTIME_GATE = '''        transfer = runner.vllm_config.kv_transfer_config
        from .offload.config import offload_options
        if offload_options(self.config) is not None and transfer is None:
            raise OscarReadinessError("OSCAR offload is enabled but native KV transfer configuration is missing")
        if transfer is not None:
            from .offload.config import validate_transfer_config, offload_options
            try:
                validate_transfer_config(transfer)
                if offload_options(self.config) != transfer.kv_connector_extra_config:
                    raise ValueError("OSCAR target and native KV transfer configuration disagree")
            except ValueError as error:
                raise OscarReadinessError(str(error)) from error
            self._offload_runner = runner
'''
_ORIGINAL_RUNTIME_GATE = '''        if runner.vllm_config.kv_transfer_config is not None:
            raise OscarReadinessError("target local-hybrid cache cannot be used by a KV transfer connector")
'''
_ALLOCATION = '''            if transfer is None:
                allocation = torch.zeros(tensor.size, dtype=torch.int8, device=runner.device)
            else:
                from .offload.config import allocate_aligned_bytes
                allocation = allocate_aligned_bytes(torch, tensor.size, runner.device)
'''
_ORIGINAL_ALLOCATION = '''            allocation = torch.zeros(tensor.size, dtype=torch.int8, device=runner.device)
'''
_LOADED_IDENTITY = '''        from .offload.config import offload_options
        if offload_options(self.config) is not None:
            from .offload.identity import rotation_artifact_identity
            self._offload_rotation_identity = rotation_artifact_identity(artifact)
'''
_CLI_EXTENSIONS = '''    from oscar_ascend.offload.config import transfer_config
    transfer = transfer_config(config)
    if transfer is not None:
        args += ["--kv-transfer-config", json.dumps(transfer, separators=(",", ":"))]
    if "enable_chunked_prefill" in config:
        chunked = config["enable_chunked_prefill"]
        if type(chunked) is not bool:
            raise ValueError("enable_chunked_prefill must be an explicit boolean")
        args += ["--enable-chunked-prefill" if chunked else "--no-enable-chunked-prefill"]
    if "offload_probe_num_gpu_blocks" in config:
        capacity = config["offload_probe_num_gpu_blocks"]
        if (os.environ.get("OSCAR_OFFLOAD_SERVICE_PROBE") != "1"
                or type(capacity) is not int or capacity <= 0):
            raise ValueError("offload cache capacity override is restricted to the explicit service probe")
        args += ["--num-gpu-blocks-override", str(capacity)]
'''
_HASH_SEED = '''    from oscar_ascend.offload.config import offload_options
    if offload_options(config) is not None:
        # Native init_none_hash reads this before building salted block hashes.
        # A fixed seed lets a later service find the same shared prefix index.
        env.setdefault("PYTHONHASHSEED", "0")
'''
_NATIVE_CONTROL = '''    if args.native:
        config = dict(config)
        config.pop("kv_offload", None)
'''
_VARIANT_EXTENSION = '''    # #127/#128: explicit transport selection never forces none -> align,
    # changes the token budget, or substitutes the native scheduler.
    from oscar_ascend.offload.config import offload_options
    if variant in {"baseline", "native"}:
        selected.pop("kv_offload", None)
    offload_options(selected)
'''


def before_authorized_offload_extension(path, text):
    if path == "configs/target.json":
        value = json.loads(text)
        assert value.pop("kv_offload") == AUTHORIZED_KV_OFFLOAD
        return json.dumps(value, indent=2) + "\n"
    changes = {
        "oscar_ascend/runtime.py": (
            (_RUNTIME_GATE, _ORIGINAL_RUNTIME_GATE), (_ALLOCATION, _ORIGINAL_ALLOCATION),
            (_LOADED_IDENTITY, "")),
        "tools/target_cli.py": ((_CLI_EXTENSIONS, ""), (_NATIVE_CONTROL, ""), (_HASH_SEED, "")),
        "tools/serving_variants.py": ((_VARIANT_EXTENSION, ""),),
    }.get(path, ())
    for current, previous in changes:
        assert text.count(current) == 1, f"{path}: authorized offload seam changed or duplicated"
        text = text.replace(current, previous, 1)
    return text
