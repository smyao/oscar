"""Explicit NPU/memcache byte-roundtrip probe; never substitutes CPU execution.

Archive #34/#36/#37--49/#127/#128/#157/#163/#171: fixed registered storage,
whole FULL/window/GDN components, and joint checkpoint completion.  This
probe validates transport contracts, not model accuracy or graph performance.
Run with torchrun --nproc-per-node=N for actual all-rank evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import traceback
from types import SimpleNamespace
import uuid

from oscar_ascend.offload.transport import (
    ByteSegment,
    MissingObjectsError,
    REGISTRATION_ALIGNMENT,
    StrictMemcacheTransport,
    TransferObject,
)
from oscar_ascend.offload.worker import CheckpointDescriptor, CheckpointWorker
from oscar_ascend.offload.metadata import MetadataClient, MetadataServer


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cleanup_remote_objects(transport, *, prefix: str, rank: int,
                            world_size: int, barrier=None) -> dict:
    """Delete this probe rank's objects only after every reader has finished."""
    if world_size > 1:
        if not callable(barrier):
            raise RuntimeError("multi-rank probe cleanup requires its completion barrier")
        barrier()
    keys = [f"{prefix}/r{rank}/full", f"{prefix}/r{rank}/gdn",
            f"{prefix}/incomplete/r{rank}/full", f"{prefix}/host/r{rank}"]
    if rank == 0:
        keys.append(f"{prefix}/manifest")
    # All these objects were created and verified by the successful probe.
    # Missing objects or deletion failures are evidence failures, not success.
    deleted = transport.host_remove_batch(keys, missing_ok=False)
    if tuple(deleted) != (True,) * len(keys):
        raise RuntimeError("probe remote objects were not all explicitly deleted")
    if any(transport.exists(keys)):
        raise RuntimeError("probe remote objects remain after deletion")
    return {"remote_object_cleanup": "passed", "remote_object_cleanup_count": len(keys)}


def _run(args: argparse.Namespace, report: dict) -> None:
    import torch
    import torch_npu  # noqa: F401; registers the real NPU backend

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("KV offload probe requires a real NPU; no CPU substitute")
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = args.local_rank
    if local_rank < 0 or local_rank >= torch.npu.device_count():
        raise RuntimeError(f"invalid local NPU rank {local_rank}")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    distributed = torch.distributed
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    run_id = args.run_id or uuid.uuid4().hex
    if world_size > 1:
        run_ids = [run_id if rank == 0 else None]
        distributed.broadcast_object_list(run_ids, src=0)
        run_id = run_ids[0]
    report.update(rank=rank, world_size=world_size, local_rank=local_rank, run_id=run_id, device=str(device))
    namespace = "oscar-kv-offload-device-probe-v1"
    prefix = f"{namespace}/{run_id}"
    keys_by_rank = {participant: (f"{prefix}/r{participant}/full", f"{prefix}/r{participant}/gdn") for participant in range(world_size)}
    descriptor = CheckpointDescriptor(namespace, run_id, keys_by_rank)

    # The owning allocation is intentionally larger than the registered
    # region, so allocator alignment is never assumed.
    owner = torch.empty(2 * REGISTRATION_ALIGNMENT, dtype=torch.uint8, device=device)
    base = int(owner.data_ptr())
    offset = (-base) % REGISTRATION_ALIGNMENT
    raw = owner[offset : offset + REGISTRATION_ALIGNMENT]
    raw.fill_(255)
    address = int(raw.data_ptr())
    if address % REGISTRATION_ALIGNMENT or offset + raw.numel() > owner.numel():
        raise RuntimeError("failed to construct an owned 2MiB-aligned NPU region")
    transport = StrictMemcacheTransport.from_vllm_config(
        SimpleNamespace(parallel_config=None), local_rank=local_rank, timeout_seconds=args.timeout_seconds,
    )
    args._resources.append(transport)
    transport.register_buffers([ByteSegment(address, raw.numel())], owners=[owner, raw])
    report["registration"] = {"alignment": REGISTRATION_ALIGNMENT, "bytes": int(raw.numel()), "explicit_status": 0}

    components = (
        ("packed", 4608), ("window_k", 512), ("window_v", 768),
        ("window_tags", 256), ("gdn_conv", 4096), ("gdn_ssm", 8192),
    )
    sources, destinations, expected, offsets = [], [], [], []
    cursor = 4096
    destination_base = 512 * 1024
    for index, (name, length) in enumerate(components):
        source_offset = cursor + index * 3
        destination_offset = destination_base + source_offset + 7
        pattern = ((torch.arange(length, dtype=torch.int64) * 17 + index * 23 + rank * 31) % 251).to(torch.uint8)
        raw[source_offset : source_offset + length].copy_(pattern.to(device))
        sources.append(ByteSegment(address + source_offset, length))
        destinations.append(ByteSegment(address + destination_offset, length))
        expected.append(pattern)
        offsets.append((name, source_offset, destination_offset, length))
        cursor += length + 257
    torch.npu.synchronize()
    local_keys = keys_by_rank[rank]
    save_objects = [TransferObject(local_keys[0], tuple(sources[:4])), TransferObject(local_keys[1], tuple(sources[4:]))]
    load_objects = [TransferObject(local_keys[0], tuple(destinations[:4])), TransferObject(local_keys[1], tuple(destinations[4:]))]

    # A deliberately incomplete, isolated generation must never be a hit.
    incomplete_keys = tuple(f"{prefix}/incomplete/r{participant}/{kind}" for participant in range(world_size) for kind in ("full", "gdn"))
    transport.put([TransferObject(f"{prefix}/incomplete/r{rank}/full", tuple(sources[:4]))])
    if world_size > 1:
        distributed.barrier()
    incomplete = transport.exists(incomplete_keys)
    if all(incomplete) or any(incomplete[index] for index in range(1, len(incomplete), 2)):
        raise RuntimeError("incomplete FULL/GDN checkpoint was incorrectly reported complete")
    report["incomplete_joint_checkpoint_rejected"] = True

    worker = CheckpointWorker(transport, participant=rank, synchronize=torch.npu.synchronize)
    local_receipt = worker.save_checkpoint(descriptor, save_objects)
    if world_size > 1:
        receipts = [None] * world_size
        distributed.all_gather_object(receipts, local_receipt)
    else:
        receipts = [local_receipt]
    worker.verify_joint_checkpoint(descriptor, receipts)
    report["joint_completion"] = {"participants": world_size, "object_count": len(descriptor.keys), "receipts": [asdict(item) for item in receipts]}

    # Overwrite sources and restore to different physical addresses.  This
    # catches an accidental no-op, source alias, or partial-component copy.
    for _, source_offset, _, length in offsets:
        raw[source_offset : source_offset + length].zero_()
    torch.npu.synchronize()
    worker.load_checkpoint(
        descriptor, load_objects,
        verify_publication=lambda checkpoint: worker.verify_joint_checkpoint(checkpoint, receipts),
    )
    torch.npu.synchronize()
    for (_, _, destination_offset, length), pattern in zip(offsets, expected):
        actual = raw[destination_offset : destination_offset + length].cpu()
        if not torch.equal(actual, pattern):
            raise RuntimeError("memcache roundtrip changed a complete byte component")

    # Missing keys must be distinguishable from SDK errors, and the local
    # destination must remain unchanged when existence fails before DMA.
    missing_target = TransferObject(f"{prefix}/missing/r{rank}", (destinations[0],))
    try:
        transport.get([missing_target])
    except MissingObjectsError:
        report["missing_key_exception"] = "MissingObjectsError"
    else:
        raise RuntimeError("missing memcache object did not raise MissingObjectsError")
    _, _, first_destination, first_length = offsets[0]
    if not torch.equal(raw[first_destination : first_destination + first_length].cpu(), expected[0]):
        raise RuntimeError("a missing-key load modified its destination")

    # These use the real documented host SDK APIs. The scheduler cannot use
    # them with init_bm=False, so its manifest traffic must traverse rank-zero
    # worker RPC. Exercise duplicate publication without assuming overwrite.
    host_key = f"{prefix}/host/r{rank}"
    host_value = f"OSCAR host metadata rank={rank} generation={run_id}".encode()
    transport.host_put(host_key, host_value)
    if transport.host_get(host_key) != host_value:
        raise RuntimeError("host memcache bytes failed exact roundtrip")
    transport.host_put(host_key, host_value, allow_duplicate=True)
    if transport.host_get(host_key) != host_value:
        raise RuntimeError("duplicate host publication changed immutable bytes")
    report["host_bytes_roundtrip"] = "exact"
    if rank == 0:
        with TemporaryDirectory(prefix="oscar-kv-probe-", dir="/tmp") as directory:
            endpoint = "ipc://" + directory + "/manifest"
            server = MetadataServer(transport, endpoint, timeout_seconds=args.timeout_seconds).start()
            client = MetadataClient(endpoint, timeout_seconds=args.timeout_seconds)
            args._resources.extend((server, client))
            index_key = f"{prefix}/manifest"
            manifest = {"namespace": namespace, "committed_tokens": 128,
                        "base_key": "prefix/data/" + run_id,
                        "index_key": index_key, "keys": list(descriptor.keys)}
            try:
                if client.get(index_key) is not None:
                    raise RuntimeError("fresh probe generation unexpectedly has a manifest")
                client.put(index_key, manifest)
                client.put(index_key, manifest)
                if client.get(index_key) != manifest:
                    raise RuntimeError("real metadata RPC changed the committed generation")
                report["metadata_rpc_roundtrip"] = "exact"
                report["duplicate_manifest_publication"] = "verified"
            finally:
                client.close()
                server.close()
                args._resources.remove(client)
                args._resources.remove(server)
    else:
        report["metadata_rpc_roundtrip"] = "rank_zero_only"
    report.update(_cleanup_remote_objects(
        transport, prefix=prefix, rank=rank, world_size=world_size,
        barrier=distributed.barrier if world_size > 1 else None,
    ))
    report.update(
        components=[{"name": name, "bytes": length, "different_destination": True} for name, length in components],
        byte_comparison="exact",
        missing_key_preserves_destination=True,
        all_rank_device_evidence=world_size > 1,
        model_accuracy="not_measured",
        graph_replay="not_measured",
        performance="not_measured",
        status="passed",
    )
    if world_size > 1:
        summaries = [None] * world_size
        distributed.all_gather_object(summaries, {
            "rank": rank, "status": report["status"],
            "remote_object_cleanup": report["remote_object_cleanup"],
            "remote_object_cleanup_count": report["remote_object_cleanup_count"],
        })
        report["rank_results"] = summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path("reports/kv_offload_probe.json"))
    parser.add_argument("--local-rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--run-id", help="optional isolated probe key namespace; defaults to a new UUID")
    args = parser.parse_args(argv)
    args._resources = []
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    output = args.output_json if world_size == 1 else args.output_json.with_name(f"{args.output_json.stem}.rank{rank}{args.output_json.suffix}")
    report = {"schema": 1, "probe": "oscar_kv_offload", "status": "running", "rank": rank, "world_size": world_size}
    _write_report(output, report)
    try:
        _run(args, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        for resource in reversed(args._resources):
            try:
                resource.close()
            except Exception as exc:
                report.update(status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
    _write_report(output, report)
    if report["status"] != "passed":
        print(f"KV OFFLOAD PROBE FAILED: {output}", file=sys.stderr, flush=True)
        return 1
    if rank == 0 and world_size > 1:
        _write_report(args.output_json, report)
    print(f"KV OFFLOAD PROBE PASSED: rank={rank}/{world_size} report={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
