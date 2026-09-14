#include "register/op_def_registry.h"

namespace ops {
class OscarInt2PagedAttention : public OpDef {
 public:
  explicit OscarInt2PagedAttention(const char* name) : OpDef(name) {
    // The 910B ccec backend cannot lower scalar bfloat16 casts used by the
    // reference kernel.  Keep the device ABI FP16; the Python adapter converts
    // BF16 model tensors at the boundary and restores the result dtype.
    // Keep literal lists here: CANN's opbuild source parser does not resolve a
    // local initializer_list variable and otherwise emits an empty dtype list.
    this->Input("qRot").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("kNew").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("vNew").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("kCache").ParamType(REQUIRED)
        .DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("vCache").ParamType(REQUIRED)
        .DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("blockTables").ParamType(REQUIRED)
        .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("qStarts").ParamType(REQUIRED)
        .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("qLens").ParamType(REQUIRED)
        .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("prefixes").ParamType(REQUIRED)
        .DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("stageK").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("stageV").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Input("owner").ParamType(REQUIRED)
        .DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).AutoContiguous();
    this->Output("attentionOut").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
    this->Attr("scaleValue").AttrType(REQUIRED).Float(1.0);
    this->Attr("numKvHeads").AttrType(REQUIRED).Int(1);
    this->Attr("headDim").AttrType(REQUIRED).Int(256);
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false);
    this->AICore().AddConfig("ascend910b", config);
  }
};
OP_ADD(OscarInt2PagedAttention);
}  // namespace ops
