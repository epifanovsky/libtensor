set(SRC_CORE
    defs.C
    exception.C
    metadata.C
    core/impl/batching_policy_base.C
    core/impl/abs_index.C
    core/impl/allocator.C
    core/impl/combined_orbits.C
    core/impl/dimensions.C
    core/impl/magic_dimensions.C
    core/impl/orbit.C
    core/impl/orbit_list.C
    core/impl/short_orbit.C
    core/impl/subgroup_orbits.C
)
set_property(SOURCE metadata.C
             APPEND PROPERTY COMPILE_DEFINITIONS BLA_VENDOR="${BLA_VENDOR}")
if (WITH_MPI)
    set_property(SOURCE metadata.C
                 APPEND PROPERTY COMPILE_DEFINITIONS WITH_MPI=1)
endif()

set(SRC_LINALG
    linalg/linalg_generic_level1.C
    linalg/linalg_generic_level2.C
    linalg/linalg_generic_level3.C
    linalg/linalg_cblas_level1.C
    linalg/linalg_cblas_level2.C
    linalg/linalg_cblas_level3.C
    linalg/BlasSequential.C
)
if (BLA_VENDOR STREQUAL "OpenBLAS")
    set_property(SOURCE linalg/BlasSequential.C
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_OPENBLAS=1
    )
elseif (BLA_VENDOR MATCHES "Intel")
    set_property(SOURCE linalg/BlasSequential.C
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_MKL=1
    )
endif()
if(HAVE_DRAND48)
    set_property(SOURCE linalg/linalg_generic_level1.C
                 APPEND PROPERTY COMPILE_DEFINITIONS HAVE_DRAND48=1)
endif()

set(SRC_KERNELS
    kernels/dadd1/kern_dadd1.C
    kernels/dadd2/kern_dadd2.C
    kernels/dcopy/kern_dcopy.C
    kernels/ddiv1/kern_ddiv1.C
    kernels/ddiv2/kern_ddiv2.C
    kernels/ddivadd1/kern_ddivadd1.C
    kernels/dmul1/kern_dmul1.C
    kernels/dmul2/kern_dmul2.C
    kernels/dmuladd1/kern_dmuladd1.C
)

set(SRC_INST
    dense_tensor/impl/dense_tensor.C
    dense_tensor/impl/to_contract2_dims.C
    dense_tensor/impl/to_diag_dims.C
    dense_tensor/impl/to_dirsum_dims.C
    dense_tensor/impl/to_ewmult2_dims.C
    dense_tensor/impl/tod_add.C
    dense_tensor/impl/tod_btconv.C
    dense_tensor/impl/tod_compare.C
    dense_tensor/impl/tod_contract2_1.C
    dense_tensor/impl/tod_contract2_2.C
    dense_tensor/impl/tod_contract2_3.C
    dense_tensor/impl/tod_contract2_4.C
    dense_tensor/impl/tod_contract2_5.C
    dense_tensor/impl/tod_contract2_6.C
    dense_tensor/impl/tod_contract2_7.C
    dense_tensor/impl/tod_contract2_8.C
    dense_tensor/impl/tod_copy.C
    dense_tensor/impl/tod_copy_wnd.C
    dense_tensor/impl/tod_diag.C
    dense_tensor/impl/tod_dirsum.C
    dense_tensor/impl/tod_dotprod.C
    dense_tensor/impl/tod_ewmult2.C
    dense_tensor/impl/tod_extract.C
    dense_tensor/impl/tod_import_raw.C
    dense_tensor/impl/tod_import_raw_stream.C
    dense_tensor/impl/tod_mult.C
    dense_tensor/impl/tod_mult1.C
    dense_tensor/impl/tod_random.C
    dense_tensor/impl/tod_scale.C
    dense_tensor/impl/tod_scatter.C
    dense_tensor/impl/tod_screen.C
    dense_tensor/impl/tod_set.C
    dense_tensor/impl/tod_set_diag.C
    dense_tensor/impl/tod_set_elem.C
    dense_tensor/impl/tod_size.C
    dense_tensor/impl/tod_trace.C
    dense_tensor/impl/tod_vmpriority.C
    symmetry/point_group_table.C
    symmetry/product_table_container.C
    symmetry/product_table_i.C
    symmetry/inst/adjacency_list.C
    symmetry/inst/block_labeling_inst.C
    symmetry/inst/combine_part_inst.C
    symmetry/inst/combine_label_inst.C
    symmetry/inst/er_merge_inst.C
    symmetry/inst/er_optimize_inst.C
    symmetry/inst/er_reduce_inst.C
    symmetry/inst/evaluation_rule_inst.C
    symmetry/inst/permutation_group_inst.C
    symmetry/inst/print_symmetry_inst.C
    symmetry/inst/se_label_inst.C
    symmetry/inst/se_part_inst.C
    symmetry/inst/se_perm_inst.C
    symmetry/inst/so_apply_inst.C
    symmetry/inst/so_apply_se_label_inst.C
    symmetry/inst/so_apply_se_part_inst.C
    symmetry/inst/so_apply_se_perm_inst.C
    symmetry/inst/so_copy_inst.C
    symmetry/inst/so_dirprod_inst.C
    symmetry/inst/so_dirprod_se_label_inst.C
    symmetry/inst/so_dirprod_se_part_inst.C
    symmetry/inst/so_dirprod_se_perm_inst.C
    symmetry/inst/so_dirsum_inst.C
    symmetry/inst/so_dirsum_se_label_inst.C
    symmetry/inst/so_dirsum_se_part_inst.C
    symmetry/inst/so_dirsum_se_perm_inst.C
    symmetry/inst/so_merge_inst.C
    symmetry/inst/so_merge_se_label_inst.C
    symmetry/inst/so_merge_se_part_inst.C
    symmetry/inst/so_merge_se_perm_inst.C
    symmetry/inst/so_permute_inst.C
    symmetry/inst/so_permute_se_label_inst.C
    symmetry/inst/so_permute_se_part_inst.C
    symmetry/inst/so_permute_se_perm_inst.C
    symmetry/inst/so_reduce_inst.C
    symmetry/inst/so_reduce_se_label_inst.C
    symmetry/inst/so_reduce_se_part_inst.C
    symmetry/inst/so_reduce_se_perm_inst.C
    symmetry/inst/so_symmetrize_inst.C
    symmetry/inst/so_symmetrize_se_label_inst.C
    symmetry/inst/so_symmetrize_se_part_inst.C
    symmetry/inst/so_symmetrize_se_perm_inst.C
    btod/btod_diagonalize.C
    btod/btod_tridiagonalize.C
)

set(SRC_GEN_BTOD
    gen_block_tensor/impl/auto_rwlock.C
    gen_block_tensor/impl/gen_bto_contract2_bis.C
)

set(SRC_BTOD
    block_tensor/impl/block_tensor.C
    block_tensor/impl/btod_addition_schedule.C
    block_tensor/impl/btod_add.C
    block_tensor/impl/btod_aux_add.C
    block_tensor/impl/btod_aux_chsym.C
    block_tensor/impl/btod_aux_copy.C
    block_tensor/impl/btod_aux_symmetrize.C
    block_tensor/impl/btod_aux_transform.C
    block_tensor/impl/btod_compare.C
    block_tensor/impl/btod_contract2_1.C
    block_tensor/impl/btod_contract2_2.C
    block_tensor/impl/btod_contract2_3.C
    block_tensor/impl/btod_contract2_4.C
    block_tensor/impl/btod_contract2_5.C
    block_tensor/impl/btod_contract2_6.C
    block_tensor/impl/btod_contract2_7.C
    block_tensor/impl/btod_contract2_8.C
    block_tensor/impl/btod_contract2_clst_builder.C
    block_tensor/impl/btod_contract2_clst_optimize.C
    block_tensor/impl/btod_contract2_nzorb.C
    block_tensor/impl/btod_contract3.C
    block_tensor/impl/btod_copy.C
    block_tensor/impl/btod_diag.C
    block_tensor/impl/btod_dirsum.C
    block_tensor/impl/btod_dotprod.C
    block_tensor/impl/btod_ewmult2.C
    block_tensor/impl/btod_export.C
    block_tensor/impl/btod_extract.C
    block_tensor/impl/btod_mult.C
    block_tensor/impl/btod_mult1.C
    block_tensor/impl/btod_random.C
    block_tensor/impl/btod_scale.C
    block_tensor/impl/btod_set_diag.C
    block_tensor/impl/btod_set_elem.C
    block_tensor/impl/btod_set.C
    block_tensor/impl/btod_shift_diag.C
    block_tensor/impl/btod_size.C
    block_tensor/impl/btod_sum.C
    block_tensor/impl/btod_symcontract3.C
    block_tensor/impl/btod_symmetrize2.C
    block_tensor/impl/btod_symmetrize3.C
    block_tensor/impl/btod_symmetrize4.C
    block_tensor/impl/btod_trace.C
    block_tensor/impl/btod_unfold_block_list.C
    block_tensor/impl/btod_unfold_symmetry.C
    block_tensor/impl/btod_vmpriority.C
)

set(SRC_EXPR
    expr/btensor/impl/eval_btensor_double.C
    expr/btensor/impl/eval_btensor_double_add.C
    expr/btensor/impl/eval_btensor_double_autoselect.C
    expr/btensor/impl/eval_btensor_double_contract.C
    expr/btensor/impl/eval_btensor_double_copy.C
    expr/btensor/impl/eval_btensor_double_diag.C
    expr/btensor/impl/eval_btensor_double_dirsum.C
    expr/btensor/impl/eval_btensor_double_div.C
    expr/btensor/impl/eval_btensor_double_dot_product.C
    expr/btensor/impl/eval_btensor_double_scale.C
    expr/btensor/impl/eval_btensor_double_set.C
    expr/btensor/impl/eval_btensor_double_symm.C
    expr/btensor/impl/eval_btensor_double_trace.C
    expr/btensor/impl/eval_tree_builder_btensor.C
    expr/btensor/impl/node_interm.C
    expr/dag/expr_tree.C
    expr/dag/graph.C
    expr/dag/node_add.C
    expr/dag/node_assign.C
    expr/dag/node_const_scalar.C
    expr/dag/node_contract.C
    expr/dag/node_diag.C
    expr/dag/node_dirsum.C
    expr/dag/node_div.C
    expr/dag/node_dot_product.C
    expr/dag/node_ident.C
    expr/dag/node_null.C
    expr/dag/node_product.C
    expr/dag/node_scalar.C
    expr/dag/node_scale.C
    expr/dag/node_set.C
    expr/dag/node_symm.C
    expr/dag/node_trace.C
    expr/dag/node_transform.C
    expr/dag/node_unblock.C
    expr/dag/node_reblock.C
    expr/dag/print_node.C
    expr/dag/print_tree.C
    expr/eval/default_eval_selector.C
    expr/eval/eval.C
    expr/eval/eval_register.C
    expr/opt/opt_add_before_transf.C
    expr/opt/opt_merge_adjacent_add.C
    expr/opt/opt_merge_adjacent_transf.C
    expr/opt/opt_merge_equiv_ident.C
)

if(WITH_LIBXM)
    set(SRC_XM_RAW
        libxm/src/alloc.c
        libxm/src/blockspace.c
        libxm/src/contract.c
        libxm/src/dim.c
        libxm/src/scalar.c
        libxm/src/tensor.c
        libxm/src/util.c
        libxm/src/xm.c
    )
    set(SRC_XM
        ${SRC_XM_RAW}
        block_tensor/impl/btod_contract2_1_xm.C
        block_tensor/impl/btod_contract2_2_xm.C
        block_tensor/impl/btod_contract2_3_xm.C
        block_tensor/impl/btod_contract2_4_xm.C
        block_tensor/impl/btod_contract2_5_xm.C
        block_tensor/impl/btod_contract2_6_xm.C
        block_tensor/impl/btod_contract2_7_xm.C
        block_tensor/impl/btod_contract2_8_xm.C
        block_tensor/impl/btod_copy_xm.C
    )

    if (WITH_MPI)
        set_property(SOURCE ${SRC_XM_RAW}
                     APPEND PROPERTY COMPILE_DEFINITIONS XM_USE_MPI=1)
        set_property(SOURCE ${SRC_XM_RAW}
                     APPEND PROPERTY INCLUDE_DIRECTORIES ${MPI_C_INCLUDE_PATH})
        set_property(SOURCE ${SRC_XM_RAW}
                     APPEND PROPERTY COMPILE_OPTIONS ${MPI_C_COMPILE_FLAGS})
    endif()
else()
    set(SRC_XM "")
endif()

set(SRC ${SRC_LINALG} ${SRC_KERNELS} ${SRC_GEN_BTOD} ${SRC_BTOD}
    ${SRC_INST} ${SRC_EXPR} ${SRC_XM}
)

add_library(tensorlight ${SRC} ${SRC_CORE})
target_include_directories(tensorlight PRIVATE "${libtensorlight_SOURCE_DIR}" "${BLAS_INCLUDE_DIR}")
target_link_libraries(tensorlight PUBLIC ${BLAS_LAPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} util)
target_compile_definitions(tensorlight PUBLIC
	$<$<CONFIG:Debug>:libtensorlight_DEBUG>)

if (WITH_MPI)
    target_link_libraries(tensorlight PUBLIC ${MPI_C_LIBRARIES} ${MPI_C_LINK_FLAGS})
endif()
