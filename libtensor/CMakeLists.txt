set(SRC
    defs.C
    exception.C
    version.C
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
    btod/btod_diagonalize.C
    btod/btod_tridiagonalize.C
    symmetry/inst/adjacency_list.C
    symmetry/point_group_table.C
    symmetry/product_table_container.C
    symmetry/product_table_i.C
    expr/expr_tree.C
    expr/graph.C
    expr/node_dot_product.C
    expr/node_product.C
    expr/node_statics.C
    expr/print_node.C
    expr/print_tree.C
    iface/eval.C
    iface/tensor_list.C
    iface/btensor/impl/eval_btensor_double.C
    iface/btensor/impl/eval_btensor_double_contract.C
    iface/btensor/impl/eval_btensor_double_copy.C
    iface/btensor/impl/eval_btensor_double_diag.C
    iface/btensor/impl/eval_btensor_double_dirsum.C
    iface/btensor/impl/eval_btensor_double_div.C
    iface/btensor/impl/eval_btensor_double_symm.C
#    iface/btensor/impl/eval_plan_builder_btensor.C
    iface/btensor/impl/eval_tree_builder_btensor.C
    iface/btensor/impl/node_interm.C
)

set(SRC_LINALG_ACML
    linalg/acml/linalg_acml_level1.C
    linalg/acml/linalg_acml_level2.C
    linalg/acml/linalg_acml_level3.C
    linalg/acml/lapack_acml.C
)

set(SRC_LINALG_CBLAS
    linalg/cblas/linalg_cblas_level1.C
    linalg/cblas/linalg_cblas_level2.C
    linalg/cblas/linalg_cblas_level3.C
)

set(SRC_LINALG_ESSL
    linalg/essl/linalg_essl_level1.C
    linalg/essl/linalg_essl_level2.C
    linalg/essl/linalg_essl_level3.C
)

set(SRC_LINALG_MKL
    linalg/mkl/linalg_mkl_level1.C
    linalg/mkl/linalg_mkl_level2.C
    linalg/mkl/linalg_mkl_level3.C
    linalg/mkl/lapack_mkl.C
)

set(SRC_LINALG_QCHEM
    linalg/qchem/linalg_qchem_level1.C
    linalg/qchem/linalg_qchem_level2.C
    linalg/qchem/linalg_qchem_level3.C
)

set(SRC_LINALG_GENERIC
    linalg/generic/linalg_generic_level1.C
    linalg/generic/linalg_generic_level2.C
    linalg/generic/linalg_generic_level3.C
)

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
    dense_tensor/impl/tod_add.C
    dense_tensor/impl/tod_btconv.C
    dense_tensor/impl/tod_compare.C
    dense_tensor/impl/tod_contract2.C
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
    block_tensor/btod/impl/btod_sum.C
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
    block_tensor/impl/btod_set_diag.C
    block_tensor/impl/btod_set_elem.C
    block_tensor/impl/btod_set.C
    block_tensor/impl/btod_size.C
    block_tensor/impl/btod_symcontract3.C
    block_tensor/impl/btod_symmetrize2.C
    block_tensor/impl/btod_symmetrize3.C
    block_tensor/impl/btod_symmetrize4.C
    block_tensor/impl/btod_trace.C
    block_tensor/impl/btod_unfold_block_list.C
    block_tensor/impl/btod_unfold_symmetry.C
    block_tensor/impl/btod_vmpriority.C
)

set(SRC_DIAG_TENSOR
    diag_block_tensor/impl/diag_block_tensor.C
    diag_block_tensor/impl/diag_btod_addition_schedule.C
    diag_block_tensor/impl/diag_btod_aux_add.C
    diag_block_tensor/impl/diag_btod_aux_copy.C
    diag_block_tensor/impl/diag_btod_contract2.C
    diag_block_tensor/impl/diag_btod_contract2_clst_builder.C
    diag_block_tensor/impl/diag_btod_copy.C
    diag_block_tensor/impl/diag_btod_dotprod.C
    diag_block_tensor/impl/diag_btod_random.C
    diag_block_tensor/impl/diag_btod_unfold_block_list.C
    diag_block_tensor/impl/tod_conv_diag_block_tensor.C
    diag_tensor/impl/diag_tensor_space.C
    diag_tensor/impl/diag_tensor.C
    diag_tensor/impl/tod_conv_diag_tensor.C
    diag_tensor/impl/diag_to_add_space.C
    diag_tensor/impl/diag_to_contract2_space.C
    diag_tensor/impl/diag_tod_adjust_space.C
    diag_tensor/impl/diag_tod_aux_constr_add.C
    diag_tensor/impl/diag_tod_aux_constr_base.C
    diag_tensor/impl/diag_tod_aux_dotprod.C
    diag_tensor/impl/diag_tod_contract2.C
    diag_tensor/impl/diag_tod_contract2s.C
    diag_tensor/impl/diag_tod_copy.C
    diag_tensor/impl/diag_tod_dotprod.C
    diag_tensor/impl/diag_tod_mult1.C
    diag_tensor/impl/diag_tod_random.C
    diag_tensor/impl/diag_tod_set.C
)

if(WITH_ARMADILLO)
    set(SRC_ARMA
        arma/impl/btod_export_arma.C
    )
else(WITH_ARMADILLO)
    set(SRC_ARMA "")
endif(WITH_ARMADILLO)

if(USE_ACML)
    set(SRC_LINALG ${SRC_LINALG_GENERIC} ${SRC_LINALG_ACML})
elseif(USE_CBLAS OR USE_GSL)
    set(SRC_LINALG ${SRC_LINALG_GENERIC} ${SRC_LINALG_CBLAS})
elseif(USE_ESSL)
    set(SRC_LINALG ${SRC_LINALG_GENERIC} ${SRC_LINALG_ESSL})
elseif(USE_MKL)
    set(SRC_LINALG ${SRC_LINALG_GENERIC} ${SRC_LINALG_MKL})
else(USE_ACML)
    set(SRC_LINALG ${SRC_LINALG_GENERIC})
endif(USE_ACML)

add_library(tensor ${SRC} ${SRC_LINALG} ${SRC_KERNELS} ${SRC_GEN_BTOD}
    ${SRC_BTOD} ${SRC_DIAG_TENSOR} ${SRC_INST} ${SRC_MP} ${SRC_ARMA})
if(QCHEM_SOURCE_DIR)
    add_library(tensor_qchem ${SRC_LINALG_GENERIC} ${SRC_LINALG_QCHEM})
    add_dependencies(tensor_qchem tensor)
endif(QCHEM_SOURCE_DIR)

set_property(TARGET tensor APPEND PROPERTY
    COMPILE_DEFINITIONS_DEBUG LIBTENSOR_DEBUG)
if(HAVE_LIBVMM)
    set_property(TARGET tensor APPEND PROPERTY
        COMPILE_DEFINITIONS_DEBUG LIBVMM_DEBUG)
else(HAVE_LIBVMM)
    set_property(TARGET tensor APPEND PROPERTY
        COMPILE_DEFINITIONS WITHOUT_LIBVMM)
endif(HAVE_LIBVMM)

