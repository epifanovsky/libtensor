#ifndef LIBTENSOR_LIBTENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "abs_index_test.h"
#include "anon_eval_test.h"
#include "bispace_test.h"
#include "bispace_expr_test.h"
#include "block_index_space_test.h"
#include "block_map_test.h"
#include "block_tensor_test.h"
#include "btensor_test.h"
#include "btod_add_test.h"
#include "btod_compare_test.h"
#include "btod_contract2_test.h"
#include "btod_copy_test.h"
#include "btod_dotprod_test.h"
#include "btod_import_raw_test.h"
#include "btod_mkdelta_test.h"
#include "btod_mult_test.h"
#include "btod_random_test.h"
#include "btod_read_test.h"
#include "btod_scale_test.h"
#include "btod_set_diag_test.h"
#include "btod_sum_test.h"
#include "contract_test.h"
#include "contraction2_test.h"
#include "contraction2_list_builder_test.h"
#include "dimensions_test.h"
#include "direct_block_tensor_test.h"
#include "direct_btensor_test.h"
#include "direct_product_test.h"
#include "dot_product_test.h"
#include "global_timings_test.h"
#include "immutable_test.h"
#include "index_test.h"
#include "index_range_test.h"
#include "labeled_btensor_test.h"
#include "letter_expr_test.h"
#include "letter_test.h"
#include "mask_test.h"
#include "orbit_test.h"
#include "orbit_list_test.h"
#include "permutation_test.h"
#include "permutation_builder_test.h"
#include "permutation_group_test.h"
#include "sequence_test.h"
#include "se_perm_test.h"
#include "so_projdown_test.h"
#include "so_projup_test.h"
#include "so_intersection_impl_perm_test.h"
#include "so_proj_down_impl_perm_test.h"
#include "so_proj_up_impl_perm_test.h"
#include "so_union_impl_perm_test.h"
#include "symel_cycleperm_test.h"
#include "symm_test.h"
#include "symmetry_test.h"
#include "symmetry_element_base_test.h"
#include "symmetry_element_set_test.h"
#include "symmetry_element_set_adapter_test.h"
#include "tensor_test.h"
#include "timer_test.h"
#include "timings_test.h"
#include "tod_add_test.h"
#include "tod_btconv_test.h"
#include "tod_compare_test.h"
#include "tod_contract2_test.h"
#include "tod_copy_test.h"
#include "tod_delta_denom2_test.h"
#include "tod_diag_test.h"
#include "tod_dotprod_test.h"
#include "tod_import_raw_test.h"
#include "tod_mkdelta_test.h"
#include "tod_mult_test.h"
#include "tod_random_test.h"
#include "tod_scale_test.h"
#include "tod_set_test.h"
#include "tod_set_diag_test.h"
#include "tod_sum_test.h"
#include "tod_symcontract2_test.h"
#include "version_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**
	\brief Test suite for the tensor library (libtensor)

	This suite runs the following tests:
	\li libtensor::abs_index_test
	\li libtensor::anon_eval_test
	\li libtensor::bispace_test
	\li libtensor::bispace_expr_test
	\li libtensor::block_index_space_test
	\li libtensor::block_map_test
	\li libtensor::block_tensor_test
	\li libtensor::btensor_test
	\li libtensor::btod_add_test
	\li libtensor::btod_compare_test
	\li libtensor::btod_contract2_test
	\li libtensor::btod_copy_test
	\li libtensor::btod_dotprod_test
	\li libtensor::btod_import_raw_test
	\li libtensor::btod_mult_test
	\li libtensor::btod_mkdelta_test
	\li libtensor::btod_random_test
	\li libtensor::btod_read_test
	\li libtensor::btod_scale_test
	\li libtensor::btod_set_diag_test
	\li libtensor::btod_sum_test
	\li libtensor::contract_test
	\li libtensor::contraction2_test
	\li libtensor::contraction2_list_builder_test
	\li libtensor::dimensions_test
	\li libtensor::direct_block_tensor_test
	\li libtensor::direct_btensor_test
	\li libtensor::direct_product_test
	\li libtensor::dot_product_test
	\li libtensor::global_timings_test
	\li libtensor::immutable_test
	\li libtensor::index_test
	\li libtensor::index_range_test
	\li libtensor::labeled_btensor_test
	\li libtensor::letter_test
	\li libtensor::letter_expr_test
	\li libtensor::mask_test
	\li libtensor::orbit_test
	\li libtensor::orbit_list_test
	\li libtensor::permutation_test
	\li libtensor::permutation_builder_test
	\li libtensor::permutation_group_test
	\li libtensor::sequence_test
	\li libtensor::se_perm_test
	\li libtensor::so_projdown_test
	\li libtensor::so_projup_test
	\li libtensor::so_intersection_impl_perm_test
	\li libtensor::so_proj_down_impl_perm_test
	\li libtensor::so_proj_up_impl_perm_test
	\li libtensor::so_union_impl_perm_test
	\li libtensor::symel_cycleperm_test
	\li libtensor::symm_test
	\li libtensor::symmetry_test
	\li libtensor::symmetry_element_base_test
	\li libtensor::symmetry_element_set_test
	\li libtensor::symmetry_element_set_adapter_test
	\li libtensor::tensor_test
	\li libtensor::timer_test
	\li libtensor::timings_test
	\li libtensor::tod_add_test
	\li libtensor::tod_btconv_test
	\li libtensor::tod_compare_test
	\li libtensor::tod_contract2_test
	\li libtensor::tod_copy_test
	\li libtensor::tod_delta_denom2_test
	\li libtensor::tod_diag_test
	\li libtensor::tod_dotprod_test
	\li libtensor::tod_import_raw_test
	\li libtensor::tod_mkdelta_test
	\li libtensor::tod_mult_test
	\li libtensor::tod_random_test
	\li libtensor::tod_scale_test
	\li libtensor::tod_set_test
	\li libtensor::tod_set_diag_test
	\li libtensor::tod_sum_test
	\li libtensor::tod_symcontract2_test
	\li libtensor::version_test
**/
class libtensor_suite : public libtest::test_suite {
private:
	unit_test_factory<abs_index_test> m_utf_abs_index;
	unit_test_factory<anon_eval_test> m_utf_anon_eval;
	unit_test_factory<bispace_test> m_utf_bispace;
	unit_test_factory<bispace_expr_test> m_utf_bispace_expr;
	unit_test_factory<block_index_space_test> m_utf_block_index_space;
	unit_test_factory<block_map_test> m_utf_block_map;
	unit_test_factory<block_tensor_test> m_utf_block_tensor;
	unit_test_factory<btensor_test> m_utf_btensor;
	unit_test_factory<btod_add_test> m_utf_btod_add;
	unit_test_factory<btod_compare_test> m_utf_btod_compare;
	unit_test_factory<btod_contract2_test> m_utf_btod_contract2;
	unit_test_factory<btod_copy_test> m_utf_btod_copy;
	unit_test_factory<btod_dotprod_test> m_utf_btod_dotprod;
	unit_test_factory<btod_import_raw_test> m_utf_btod_import_raw;
	unit_test_factory<btod_mkdelta_test> m_utf_btod_mkdelta;
	unit_test_factory<btod_mult_test> m_utf_btod_mult;
	unit_test_factory<btod_random_test> m_utf_btod_random;
	unit_test_factory<btod_read_test> m_utf_btod_read;
	unit_test_factory<btod_scale_test> m_utf_btod_scale;
	unit_test_factory<btod_set_diag_test> m_utf_btod_set_diag;
	unit_test_factory<btod_sum_test> m_utf_btod_sum;
	unit_test_factory<contract_test> m_utf_contract;
	unit_test_factory<contraction2_test> m_utf_contraction2;
	unit_test_factory<contraction2_list_builder_test>
		m_utf_contraction2_list_builder;
	unit_test_factory<dimensions_test> m_utf_dimensions;
	unit_test_factory<direct_block_tensor_test> m_utf_direct_block_tensor;
	unit_test_factory<direct_btensor_test> m_utf_direct_btensor;
	unit_test_factory<direct_product_test> m_utf_direct_product;
	unit_test_factory<dot_product_test> m_utf_dot_product;
	unit_test_factory<global_timings_test> m_utf_global_timings;
	unit_test_factory<immutable_test> m_utf_immutable;
	unit_test_factory<index_test> m_utf_index;
	unit_test_factory<index_range_test> m_utf_index_range;
	unit_test_factory<labeled_btensor_test> m_utf_labeled_btensor;
	unit_test_factory<letter_test> m_utf_letter;
	unit_test_factory<letter_expr_test> m_utf_letter_expr;
	unit_test_factory<mask_test> m_utf_mask;
	unit_test_factory<orbit_test> m_utf_orbit;
	unit_test_factory<orbit_list_test> m_utf_orbit_list;
	unit_test_factory<permutation_test> m_utf_permutation;
	unit_test_factory<permutation_builder_test> m_utf_permutation_builder;
	unit_test_factory<permutation_group_test> m_utf_permutation_group;
	unit_test_factory<sequence_test> m_utf_sequence;
	unit_test_factory<se_perm_test> m_utf_se_perm;
	unit_test_factory<so_projdown_test> m_utf_so_projdown;
	unit_test_factory<so_projup_test> m_utf_so_projup;
	unit_test_factory<so_intersection_impl_perm_test>
		m_utf_so_intersection_impl_perm;
	unit_test_factory<so_proj_down_impl_perm_test>
		m_utf_so_proj_down_impl_perm;
	unit_test_factory<so_proj_up_impl_perm_test> m_utf_so_proj_up_impl_perm;
	unit_test_factory<so_union_impl_perm_test> m_utf_so_union_impl_perm;
	unit_test_factory<symel_cycleperm_test> m_utf_symel_cycleperm;
	unit_test_factory<symm_test> m_utf_symm;
	unit_test_factory<symmetry_test> m_utf_symmetry;
	unit_test_factory<symmetry_element_base_test>
		m_utf_symmetry_element_base;
	unit_test_factory<symmetry_element_set_test> m_utf_symmetry_element_set;
	unit_test_factory<symmetry_element_set_adapter_test>
		m_utf_symmetry_element_set_adapter;
	unit_test_factory<tensor_test> m_utf_tensor;
	unit_test_factory<timer_test> m_utf_timer;
	unit_test_factory<timings_test> m_utf_timings;
	unit_test_factory<tod_add_test> m_utf_tod_add;
	unit_test_factory<tod_btconv_test> m_utf_tod_btconv;
	unit_test_factory<tod_compare_test> m_utf_tod_compare;
	unit_test_factory<tod_contract2_test> m_utf_tod_contract2;
	unit_test_factory<tod_copy_test> m_utf_tod_copy;
	unit_test_factory<tod_delta_denom2_test> m_utf_tod_delta_denom2;
	unit_test_factory<tod_diag_test> m_utf_tod_diag;
	unit_test_factory<tod_dotprod_test> m_utf_tod_dotprod;
	unit_test_factory<tod_import_raw_test> m_utf_tod_import_raw;
	unit_test_factory<tod_mkdelta_test> m_utf_tod_mkdelta;
	unit_test_factory<tod_mult_test> m_utf_tod_mult;
	unit_test_factory<tod_random_test> m_utf_tod_random;
	unit_test_factory<tod_scale_test> m_utf_tod_scale;
	unit_test_factory<tod_set_test> m_utf_tod_set;
	unit_test_factory<tod_set_diag_test> m_utf_tod_set_diag;
	unit_test_factory<tod_sum_test> m_utf_tod_sum;
	unit_test_factory<tod_symcontract2_test> m_utf_tod_symcontract2;
	unit_test_factory<version_test> m_utf_version;

public:
	//!	Creates the suite
	libtensor_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SUITE_H

