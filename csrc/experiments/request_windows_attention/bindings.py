"""Archive #17-22/#34/#36/#69/#87/#108/#155: explicit checked new ABI.

Generate a separate binding from this project's existing checked reader;
neither old bindings nor reference trees are changed. Not in deployment yet.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    text = (ROOT/'csrc/striped_attention_bindings.cpp').read_text().split('TORCH_LIBRARY_FRAGMENT')[0]
    text = text.replace('#include "include/oscar_attention_launch.h"',
                        '#include "include/oscar_attention_launch.h"\n#include "include/oscar_request_window_launch.h"')
    text = text.replace('AttentionStriped', 'AttentionRequestWindow')
    text = text.replace('const at::Tensor& windowTags,const at::Tensor& tasks,',
                        'const at::Tensor& windowTags,const at::Tensor& requestMap,const at::Tensor& tasks,')
    text = text.replace('splits,double scale,int64_t cores)',
                        'splits,double scale,int64_t cores,int64_t windowCapacity,int64_t positionBits)')
    text = text.replace('blockTokens%128==0', 'blockTokens%256==0')
    text = text.replace('pageStride>=blockTokens*hk*(d/2+8)', 'pageStride/(hk*(d/2+8))>=blockTokens/2')
    text = text.replace('blocks<=(raw.numel()-ssmOffset)/pageStride',
                        'blocks<=(raw.numel()-ssmOffset)/pageStride/2')
    text = text.replace('window.size(0)==blocks', 'window.size(0)==windowCapacity')
    text = text.replace('windowTags.size(0)==blocks', 'windowTags.size(0)==windowCapacity')
    text = text.replace('  const int64_t windowRows=sink+recent+speculative;', '''  TORCH_CHECK(windowCapacity>0 && windowCapacity<=4096 && positionBits>0 &&
      positionBits<63,"invalid stable-window capacity or position bits");
  Check(requestMap,query,at::kLong,"request_map");
  TORCH_CHECK(requestMap.dim()==2 && requestMap.size(0)==table.size(0) &&
      requestMap.size(0)<=windowCapacity && requestMap.size(1)==2,
      "request_map must be int64[metadata_requests,2]");
  const int64_t windowRows=sink+recent+speculative;''')
    text = text.replace('windowKey,windowValue,windowTags,tasks}',
                        'windowKey,windowValue,windowTags,requestMap,tasks}')
    text = text.replace('windowValue.data_ptr(),windowTags.data_ptr(),tasks.data_ptr(),',
                        'windowValue.data_ptr(),windowTags.data_ptr(),requestMap.data_ptr(),tasks.data_ptr(),')
    text = text.replace('static_cast<float>(scale),static_cast<uint32_t>(cores)',
                        'static_cast<float>(scale),windowCapacity,positionBits,static_cast<uint32_t>(cores)')
    specs = [('', False, False), ('_balanced', False, False),
             ('_decode', False, True), ('_decode_q1', False, True),
             ('_cluster4', True, False), ('_cluster16', True, False)]
    text += 'TORCH_LIBRARY_FRAGMENT(oscar_ascend_ops,m) {\n'
    for suffix, cluster, _ in specs:
        schema = ('Tensor query, Tensor query_rot, Tensor current_key, Tensor current_value, '
                  'Tensor rotation_v, Tensor raw, Tensor block_table, Tensor window_key, Tensor window_value, '
                  'Tensor window_tags, Tensor request_map, Tensor tasks, Tensor(a!) partial, Tensor(b!) lse, '
                  'Tensor(c!) status, Tensor(d!) workspace, ')
        if cluster: schema += 'Tensor(e!) cluster_stats, '
        schema += ('int block_tokens, int physical_blocks, int raw_ssm_offset, int physical_page_stride, '
                   'int sink_tokens, int recent_tokens, int speculative_tokens, int splits, float scale, '
                   'int cube_cores, int window_capacity, int position_bits')
        text += f'  m.def("attention_request_window{suffix}_out({schema}) -> ()");\n'
    text += '}\nTORCH_LIBRARY_IMPL(oscar_ascend_ops,PrivateUse1,m) {\n'
    header = '''// Archive #17-22/#34/#36/#69/#87/#108/#155: new cache addressing ABI.
#pragma once
#include <cstdint>
namespace oscar_ascend {
'''
    for suffix, cluster, small in specs:
        launch = 'attention_request_window'+suffix+'_launch'
        function = 'AttentionRequestWindowCluster' if cluster else 'AttentionRequestWindow'
        extra = ','+suffix.removeprefix('_cluster') if cluster else (
            ',true,oscar_ascend::attention_decode_bundle_workspace_per_core(256)' if small else '')
        text += f'  m.impl("attention_request_window{suffix}_out", &{function}<oscar_ascend::{launch}{extra}>);\n'
        pointer_count = 18 if cluster else 17  # stream + 16 reader tensors [+ cluster stats]
        header += f'void {launch}('+','.join(['void*']*pointer_count)+',\n    '
        header += ','.join(['int64_t']*17)+',float,int64_t,int64_t,uint32_t);\n'
    text += '}\n'
    header += '}\n'
    (ROOT/'csrc/request_window_attention_bindings.cpp').write_text(text)
    (ROOT/'csrc/include/oscar_request_window_launch.h').write_text(header)


if __name__ == '__main__':
    main()
