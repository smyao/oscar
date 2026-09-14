add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        oscar_int2_paged_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME OscarInt2PagedAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -mllvm -cce-vf-remove-membar=false
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE oscar_int2_paged_attention ACLNNTYPE aclnn)
endif()
