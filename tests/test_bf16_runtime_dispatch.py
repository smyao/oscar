"""BF16 branch host routing/allocation contracts; never NPU numerical evidence."""
import json
from pathlib import Path

import pytest
import torch

from oscar_ascend import rotations, runtime
from oscar_ascend.integration.runtime_api import OscarReadinessError
from oscar_ascend.ops.cv_dispatch import BF16_CV_OPS
from oscar_ascend.ops.hybrid_layout import pack_rotation_nz
from tools.serving_variants import variant_config


@pytest.mark.parametrize("n,later,expected", [
    (8,False,"attention_cv_bf16_decode_out"),
    (8,True,"attention_cv_bf16_decode_q1_out"),
    (256,False,"attention_cv_bf16_cluster4_out"),
    (8192,False,"attention_cv_bf16_cluster16_out"),
    (10,False,"attention_cv_bf16_cluster4_out"),
])
def test_candidate_reaches_real_forward_reader_and_keeps_writer_fp32(n,later,expected):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        n,1,1,n,16,later,True,True,True,_member_pipeline=True,_c16_l1=True,
        _bf16_expected=expected)


@pytest.mark.parametrize("bad", ["missing","float32","float32_nz"])
def test_forward_fails_closed_before_cv_when_bf16_rv_missing_or_wrong_dtype(bad):
    from test_runtime import test_forward_dispatches_native_draft_strides_and_int32_slots_in_order
    test_forward_dispatches_native_draft_strides_and_int32_slots_in_order(
        8,1,1,8,16,False,True,True,True,_member_pipeline=True,_c16_l1=True,
        _bf16_expected="attention_cv_bf16_decode_out",_bf16_rv=bad)


@pytest.mark.parametrize("bf16", [True,False])
def test_natural_bf16_rv_is_cached_during_workspace_reservation_without_changing_writer(
        monkeypatch,tmp_path,bf16):
    name="model.layers.0.self_attn.attn"
    (tmp_path/"config.json").write_text(json.dumps({"num_hidden_layers":1,"layer_types":["full_attention"],"head_dim":256}))
    config=variant_config(json.loads(Path("configs/target.json").read_text()),"candidate")
    config.update(model=str(tmp_path),experimental_bf16_attention=bf16,experimental_aligned_math=False,
                  experimental_c16_tuned=False,rotation_method="hadamard")
    provider=runtime.AscendRuntimeProvider(config)
    key=torch.eye(256);value=torch.eye(256)*2
    inverse=torch.arange(256*256,dtype=torch.float32).reshape(256,256)/4096 - 0.7
    original=inverse.clone();device_rotation=rotations.DeviceRotation(key,value,inverse,False)
    calls=[]
    monkeypatch.setattr(rotations,"load_active_artifact",lambda *args,**kwargs: calls.append("load") or object())
    def prepare(_artifact,*,device):
        assert device=="cpu";calls.append("prepare")
        return {name:device_rotation}
    monkeypatch.setattr(rotations,"prepare_device_rotations",prepare)
    monkeypatch.setattr(provider,"_device_cube_cores",lambda _device:20)
    class Workspace:
        def __init__(self,geometry,device):
            assert device=="cpu";self.geometry=geometry;calls.append("workspace")
    monkeypatch.setattr(runtime,"GraphWorkspace",Workspace)
    workspace=provider.ensure_workspace(6,1,256,"cpu")
    assert calls==["workspace","load","prepare"]
    assert workspace.geometry.cv_bytes==20*7487488
    cached=provider.bf16_rotations if bf16 else provider.hybrid_rotations
    matrix=cached[("cpu",256)][name]
    if bf16:
        assert matrix.dtype==torch.bfloat16 and matrix.is_contiguous()
        assert torch.equal(matrix,inverse.to(torch.bfloat16))
        assert not torch.equal(matrix,pack_rotation_nz(inverse).to(torch.bfloat16))
        assert not provider.hybrid_rotations
        assert cached[("cpu",256)][None].dtype==torch.bfloat16
        assert torch.equal(cached[("cpu",256)][None],torch.eye(256,dtype=torch.bfloat16))
    else:
        assert matrix.dtype==torch.float32
        assert torch.equal(matrix,pack_rotation_nz(inverse))
        assert not provider.bf16_rotations
    assert matrix.data_ptr()!=inverse.data_ptr()
    assert provider.rotations[("cpu",256)][name] is device_rotation
    assert device_rotation.key_transposed is key and device_rotation.value_transposed is value
    assert inverse.dtype==torch.float32 and torch.equal(inverse,original)
    assert provider.ensure_workspace(6,1,256,"cpu") is workspace
    assert provider._prepare_rotations(256,"cpu")[name] is device_rotation
    assert cached[("cpu",256)][name] is matrix
    assert calls==["workspace","load","prepare"]


@pytest.mark.parametrize("missing", sorted(BF16_CV_OPS))
def test_missing_bf16_capability_fails_before_npu_initialization(monkeypatch,missing):
    from oscar_ascend.ops import loader
    config=variant_config(json.loads(Path("configs/target.json").read_text()),"candidate")
    config["experimental_aligned_math"]=False
    config["experimental_c16_tuned"]=False
    provider=runtime.AscendRuntimeProvider(config)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES",",".join(map(str,config["devices"])))
    monkeypatch.setattr(loader,"require_production_ops",lambda:None)
    calls=[]
    def require(names):
        calls.append(set(names))
        if missing in names:raise RuntimeError("missing BF16 reader")
    monkeypatch.setattr(loader,"require_capabilities",require)
    with pytest.raises(RuntimeError,match="missing BF16 reader"):
        provider.assert_ready()
    assert calls==[set(BF16_CV_OPS)] and provider._ready is False


def test_bf16_rotation_preallocation_rejects_non_d256_before_loading(monkeypatch):
    config=variant_config(json.loads(Path("configs/target.json").read_text()),"candidate")
    provider=runtime.AscendRuntimeProvider(config)
    monkeypatch.setattr(rotations,"load_artifact",lambda *a,**k:pytest.fail("artifact loaded before geometry check"))
    with pytest.raises(OscarReadinessError,match="D256"):
        provider._prepare_rotations(128,"cpu")
