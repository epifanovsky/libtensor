set(TESTS
    abs_index_test
    block_index_space_product_builder_test
    block_index_space_test
    block_index_subspace_builder_test
    block_map_test
    combined_orbits_test
    contraction2_list_builder_test
    contraction2_test
    dimensions_test
    immutable_test
    index_range_test
    index_test
    magic_dimensions_test
    mask_test
    orbit_list_test
    orbit_test
    permutation_builder_test
    permutation_generator_test
    permutation_test
    sequence_generator_test
    sequence_test
    short_orbit_test
    subgroup_orbits_test
    symmetry_element_set_test
    symmetry_test
    transf_list_test
    version_test
)

libtensor_add_tests(core ${TESTS})

