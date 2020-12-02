set(TESTS
    dense_tensor_test
    to_contract2_dims_test
    tod_add_test
    tod_apply_test
    tod_btconv_test
    tod_compare_test
    tod_contract2_test
    # tod_copy_test
    tod_copy_wnd_test
    tod_diag_test
    tod_dirsum_test
    tod_dotprod_test
    tod_ewmult2_test
    tod_extract_test
    tod_import_raw_stream_test
    tod_import_raw_test
    tod_mult1_test
    tod_mult_test
    tod_random_test
    tod_scale_test
    tod_scatter_test
    tod_select_test
    tod_set_diag_test
    tod_set_elem_test
    tod_set_test
    tod_size_test
    tod_trace_test
    tod_vmpriority_test
)

libtensor_add_tests(dense_tensor ${TESTS})

