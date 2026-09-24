// SPDX-License-Identifier: Apache-2.0
// Archive G5/G6/#34/#36/#53/#58/#69/#92/#108: explicit same-device schema,
// fixed graph capacities and validated strided native page views.
// D.4: prepare and fused attention; one metadata launch and one CV launch,
// fixed per-Cube tile workspace, no full-history tensor or host request loop.
// Previous 725ms host prepare / 6499.8-6655.1ms dequant must be measured anew;
// source presence and successful compilation are not numerical/performance proof.
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include "include/oscar_attention_launch.h"
namespace {
void Check(const at::Tensor& x,const at::Tensor& owner,at::ScalarType type,
           const char* name,bool contiguous=true) {
  TORCH_CHECK(x.device().type()==c10::DeviceType::PrivateUse1 &&
      x.device()==owner.device(),name," must share the input NPU");
  TORCH_CHECK(x.scalar_type()==type,name," has incorrect dtype");
  TORCH_CHECK(!contiguous || x.is_contiguous(),name," must be contiguous");
}
void Disjoint(const at::Tensor& a,const at::Tensor& b) {
  if(!a.numel() || !b.numel()) return;
  const uintptr_t ap=reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t bp=reinterpret_cast<uintptr_t>(b.data_ptr());
  uint64_t an=a.element_size(),bn=b.element_size();
  for(int64_t axis=0;axis<a.dim();++axis)
    an+=static_cast<uint64_t>(a.size(axis)-1)*a.stride(axis)*a.element_size();
  for(int64_t axis=0;axis<b.dim();++axis)
    bn+=static_cast<uint64_t>(b.size(axis)-1)*b.stride(axis)*b.element_size();
  TORCH_CHECK(ap<bp?bp-ap>=an:ap-bp>=bn,"attention output aliases an input/output");
}
void Prepare(const at::Tensor& starts,const at::Tensor& lens,const at::Tensor& slots,
             at::Tensor tasks,at::Tensor positions,int64_t hq,int64_t hk,int64_t sink,int64_t recent,
             int64_t splits,const std::optional<at::Tensor>& blockTable,bool slotContext) {
  Check(starts,starts,at::kInt,"qstarts");Check(lens,starts,at::kInt,"seq_lens");
  TORCH_CHECK(slots.scalar_type()==at::kInt || slots.scalar_type()==at::kLong,
      "slot_mapping must be int32/int64");
  Check(slots,starts,slots.scalar_type(),"slot_mapping");
  Check(tasks,starts,at::kLong,"tasks");Check(positions,starts,at::kLong,"positions");
  TORCH_CHECK(positions.dim()==1 && positions.numel()==slots.numel(),"positions must be int64[N]");
  TORCH_CHECK(starts.dim()==1 && lens.dim()==1 && lens.numel()>0 &&
      lens.numel()<=4096 && starts.numel()==lens.numel()+1,"invalid request metadata");
  TORCH_CHECK(slots.dim()==1 && hk>0 && hq>=hk && hq%hk==0 && hq/hk<=16,
      "GQA ratio must be in [1,16], so a 64-row tile reuses all four MTP queries");
  TORCH_CHECK(sink>=0 && recent>0 && splits>0 && splits<=32,"invalid window/splits");
  TORCH_CHECK(slots.numel()<=std::numeric_limits<int64_t>::max()/(hk*3*splits),
      "task count overflows");
  TORCH_CHECK(tasks.dim()==2 && tasks.size(0)==slots.numel()*hk*3*splits &&
      tasks.size(1)==16,"tasks must be int64 [N*Hkv*3*splits,16]");
  if(slotContext) {
    TORCH_CHECK(blockTable.has_value(),"slot-derived MTP context requires block_table");
    Check(*blockTable,starts,at::kInt,"block_table");
    TORCH_CHECK(blockTable->dim()==2 && blockTable->size(0)>=lens.numel() &&
        blockTable->size(1)>0,"MTP block_table must cover every request");
    Disjoint(tasks,*blockTable);Disjoint(positions,*blockTable);
  }
  Disjoint(tasks,starts);Disjoint(tasks,lens);Disjoint(tasks,slots);
  Disjoint(tasks,positions);Disjoint(positions,starts);Disjoint(positions,lens);Disjoint(positions,slots);
  if(!slots.numel()) return;
  const c10_npu::OptionalNPUGuard guard(starts.device());
  oscar_ascend::prepare_attention_tasks_launch(c10_npu::getCurrentNPUStream().stream(),
      starts.data_ptr(),lens.data_ptr(),slots.data_ptr(),tasks.data_ptr(),positions.data_ptr(),lens.numel(),
      slots.numel(),hq,hk,sink,recent,splits,slots.scalar_type()==at::kLong,
      slotContext?blockTable->data_ptr():nullptr,slotContext?blockTable->size(1):0,slotContext,32);
}
#define OSCAR_CV_BINDING_ARGS const at::Tensor& query,const at::Tensor& queryRot, \
    const at::Tensor& currentKey,const at::Tensor& currentValue, \
    const at::Tensor& rotation,const at::Tensor& raw,const at::Tensor& table, \
    const at::Tensor& windowKey,const at::Tensor& windowValue, \
    const at::Tensor& windowTags,const at::Tensor& tasks,at::Tensor partial, \
    at::Tensor lse,at::Tensor status,at::Tensor workspace,int64_t blockTokens, \
    int64_t blocks,int64_t ssmOffset,int64_t pageStride,int64_t sink, \
    int64_t recent,int64_t speculative,int64_t splits,double scale,int64_t cores
#define OSCAR_CV_BINDING_PASS query,queryRot,currentKey,currentValue,rotation,raw,table, \
    windowKey,windowValue,windowTags,tasks,partial,lse,status,workspace,blockTokens, \
    blocks,ssmOffset,pageStride,sink,recent,speculative,splits,scale,cores
void AttentionCommon(OSCAR_CV_BINDING_ARGS,const std::optional<at::Tensor>& profile) {
  Check(query,query,at::kBFloat16,"query");Check(queryRot,query,at::kFloat,"query_rot");
  Check(currentKey,query,at::kBFloat16,"current_key");
  Check(currentValue,query,at::kBFloat16,"current_value");
  Check(rotation,query,at::kFloat,"rotation_v");Check(raw,query,at::kByte,"raw");
  Check(table,query,at::kInt,"block_table");Check(tasks,query,at::kLong,"tasks");
  Check(windowKey,query,at::kBFloat16,"window_key",false);
  Check(windowValue,query,at::kBFloat16,"window_value",false);
  Check(windowTags,query,at::kLong,"window_tags",false);
  Check(partial,query,at::kFloat,"partial");Check(lse,query,at::kFloat,"lse");
  Check(status,query,at::kInt,"status");Check(workspace,query,at::kByte,"workspace");
  TORCH_CHECK(query.dim()==3 && queryRot.sizes()==query.sizes(),"Q shape mismatch");
  const auto n=query.size(0),hq=query.size(1),d=query.size(2);
  TORCH_CHECK(d==64 || d==128 || d==256,"D must be 64/128/256");
  TORCH_CHECK(currentKey.dim()==3 && currentKey.size(0)==n &&
      currentKey.size(2)==d && currentValue.sizes()==currentKey.sizes(),"current K/V shape mismatch");
  const auto hk=currentKey.size(1);
  TORCH_CHECK(hk>0 && hq>=hk && hq%hk==0 && hq/hk<=16,"invalid GQA");
  TORCH_CHECK(rotation.dim()==2 && rotation.size(0)==d && rotation.size(1)==d,
      "rotation_v must be float32 [D,D]");
  TORCH_CHECK(blocks>0 && blockTokens>0 && blockTokens%128==0 && ssmOffset>=0 &&
      pageStride>=blockTokens*hk*(d/2+8),"invalid compressed physical geometry");
  TORCH_CHECK(raw.dim()==1 && ssmOffset<=raw.numel() &&
      blocks<=(raw.numel()-ssmOffset)/pageStride,"compressed raw arena is too small");
  TORCH_CHECK(sink>=0 && recent>0 && speculative>=0 && splits>0 && splits<=32 &&
      cores>0 && cores<=32 && std::isfinite(scale) && scale>0,"invalid launch attributes");
  const int64_t windowRows=sink+recent+speculative;
  for(const auto& window:{windowKey,windowValue}) {
    TORCH_CHECK(window.dim()==4 && window.size(0)==blocks && window.size(1)==windowRows &&
        window.size(2)==hk && window.size(3)==d && window.stride(3)==1 &&
        window.stride(2)==d && window.stride(1)==hk*d &&
        window.stride(0)>=windowRows*hk*d,"invalid precise page view");
  }
  TORCH_CHECK(windowKey.stride(0)==windowValue.stride(0),"K/V page stride mismatch");
  TORCH_CHECK(windowTags.dim()==2 && windowTags.size(0)==blocks &&
      windowTags.size(1)==windowRows && windowTags.stride(1)==1 &&
      windowTags.stride(0)>=windowRows,"invalid precise tag view");
  TORCH_CHECK(table.dim()==2 && table.size(0)>0 && table.size(1)>0,"invalid block table");
  TORCH_CHECK(tasks.dim()==2 && tasks.size(0)==n*hk*3*splits && tasks.size(1)==16,
      "invalid attention task capacity");
  TORCH_CHECK(partial.dim()==4 && partial.size(0)==n && partial.size(1)==hq &&
      partial.size(2)==3*splits && partial.size(3)==d,"partial must be [N,Hq,3*splits,D]");
  TORCH_CHECK(lse.dim()==3 && lse.size(0)==n && lse.size(1)==hq &&
      lse.size(2)==3*splits,"LSE must be [N,Hq,3*splits]");
  TORCH_CHECK(status.dim()==2 && status.size(0)==tasks.size(0) && status.size(1)==2,
      "status must be [T,2], one word for each AIV");
  TORCH_CHECK(workspace.dim()==1 && workspace.numel()>=cores*
      oscar_ascend::attention_workspace_per_core(d),"bounded tile workspace too small");
  for(const auto& written:{partial,lse,status,workspace})
    for(const auto& input:{query,queryRot,currentKey,currentValue,rotation,raw,table,
                          windowKey,windowValue,windowTags,tasks}) Disjoint(written,input);
  Disjoint(partial,lse);Disjoint(partial,status);Disjoint(partial,workspace);
  Disjoint(lse,status);Disjoint(lse,workspace);Disjoint(status,workspace);
  if(profile.has_value()) {
    Check(*profile,query,at::kLong,"profile");
    TORCH_CHECK(profile->dim()==4 && profile->size(0)==cores &&
        profile->size(1)==oscar_ascend::kAttentionProfileEngines &&
        profile->size(2)==oscar_ascend::kAttentionProfileSources &&
        profile->size(3)==oscar_ascend::kAttentionProfileFields,
        "profile must be int64 [cube_cores,3,4,20]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(profile->data_ptr()) %
        oscar_ascend::kAttentionProfileCacheLineBytes==0,
        "profile buffer must be 64-byte aligned for per-owner DCache flushing");
    for(const auto& existing:{query,queryRot,currentKey,currentValue,rotation,raw,table,
                              windowKey,windowValue,windowTags,tasks,partial,lse,status,workspace})
      Disjoint(*profile,existing);
    TORCH_CHECK(n>0,"CV diagnostic profile requires at least one token");
  }
  if(!n) return;
  const c10_npu::OptionalNPUGuard guard(query.device());
  if(profile.has_value())
    oscar_ascend::attention_cv_profile_launch(c10_npu::getCurrentNPUStream().stream(),
        query.data_ptr(),queryRot.data_ptr(),currentKey.data_ptr(),currentValue.data_ptr(),
        rotation.data_ptr(),raw.data_ptr(),table.data_ptr(),windowKey.data_ptr(),
        windowValue.data_ptr(),windowTags.data_ptr(),tasks.data_ptr(),partial.data_ptr(),
        lse.data_ptr(),status.data_ptr(),workspace.data_ptr(),n,hq,hk,d,table.size(0),
        table.size(1),tasks.size(0),blockTokens,blocks,ssmOffset,pageStride,
        windowKey.stride(0),windowTags.stride(0),sink,recent,speculative,splits,
        static_cast<float>(scale),static_cast<uint32_t>(cores),profile->data_ptr());
  else
    oscar_ascend::attention_cv_launch(c10_npu::getCurrentNPUStream().stream(),
        query.data_ptr(),queryRot.data_ptr(),currentKey.data_ptr(),currentValue.data_ptr(),
        rotation.data_ptr(),raw.data_ptr(),table.data_ptr(),windowKey.data_ptr(),
        windowValue.data_ptr(),windowTags.data_ptr(),tasks.data_ptr(),partial.data_ptr(),
        lse.data_ptr(),status.data_ptr(),workspace.data_ptr(),n,hq,hk,d,table.size(0),
        table.size(1),tasks.size(0),blockTokens,blocks,ssmOffset,pageStride,
        windowKey.stride(0),windowTags.stride(0),sink,recent,speculative,splits,
        static_cast<float>(scale),static_cast<uint32_t>(cores));
}
void Attention(OSCAR_CV_BINDING_ARGS) {
  AttentionCommon(OSCAR_CV_BINDING_PASS,std::nullopt);
}
void AttentionProfile(OSCAR_CV_BINDING_ARGS,at::Tensor profile) {
  AttentionCommon(OSCAR_CV_BINDING_PASS,profile);
}
#undef OSCAR_CV_BINDING_ARGS
#undef OSCAR_CV_BINDING_PASS
}
TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {
  m.def("prepare_attention_tasks_out(Tensor query_start_loc, Tensor seq_lens, Tensor slot_mapping, "
      "Tensor(a!) tasks, Tensor(b!) positions, int query_heads, int kv_heads, int sink_tokens, int recent_tokens, int splits, Tensor? block_table=None, bool use_slot_context=False) -> ()");
  m.def("attention_cv_out(Tensor query, Tensor query_rot, Tensor current_key, Tensor current_value, "
      "Tensor rotation_v, Tensor raw, Tensor block_table, Tensor window_key, Tensor window_value, "
      "Tensor window_tags, Tensor tasks, Tensor(a!) partial, Tensor(b!) lse, Tensor(c!) status, "
      "Tensor(d!) workspace, int block_tokens, int physical_blocks, int raw_ssm_offset, "
      "int physical_page_stride, int sink_tokens, int recent_tokens, int speculative_tokens, "
      "int splits, float scale, int cube_cores) -> ()");
  m.def("attention_cv_profile_out(Tensor query, Tensor query_rot, Tensor current_key, Tensor current_value, "
      "Tensor rotation_v, Tensor raw, Tensor block_table, Tensor window_key, Tensor window_value, "
      "Tensor window_tags, Tensor tasks, Tensor(a!) partial, Tensor(b!) lse, Tensor(c!) status, "
      "Tensor(d!) workspace, int block_tokens, int physical_blocks, int raw_ssm_offset, "
      "int physical_page_stride, int sink_tokens, int recent_tokens, int speculative_tokens, "
      "int splits, float scale, int cube_cores, Tensor(e!) profile) -> ()");
}
TORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {
  m.impl("prepare_attention_tasks_out",&Prepare);m.impl("attention_cv_out",&Attention);
  m.impl("attention_cv_profile_out",&AttentionProfile);
}
