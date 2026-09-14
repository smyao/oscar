#include "register/op_def_registry.h"

namespace ops {
class OscarInt2PagedAttention : public OpDef {
 public:
  explicit OscarInt2PagedAttention(const char* name) : OpDef(name) {
    auto fp = std::initializer_list<ge::DataType>{ge::DT_FLOAT16, ge::DT_BF16};
    auto nd = std::initializer_list<ge::Format>{ge::FORMAT_ND, ge::FORMAT_ND};
    this->Input("qRot").ParamType(REQUIRED).DataType(fp).Format(nd).AutoContiguous();
    this->Input("kNew").ParamType(REQUIRED).DataType(fp).Format(nd).AutoContiguous();
    this->Input("vNew").ParamType(REQUIRED).DataType(fp).Format(nd).AutoContiguous();
    this->Input("kCache").ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8}).Format(nd).AutoContiguous();
    this->Input("vCache").ParamType(REQUIRED)
        .DataType({ge::DT_INT8, ge::DT_INT8}).Format(nd).AutoContiguous();
    this->Input("blockTables").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format(nd).AutoContiguous();
    this->Input("qStarts").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format(nd).AutoContiguous();
    this->Input("qLens").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format(nd).AutoContiguous();
    this->Input("prefixes").ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32}).Format(nd).AutoContiguous();
    this->Input("stageK").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(nd).AutoContiguous();
    this->Input("stageV").ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format(nd).AutoContiguous();
    this->Input("owner").ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64}).Format(nd).AutoContiguous();
    this->Output("attentionOut").ParamType(REQUIRED).DataType(fp).Format(nd);
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
