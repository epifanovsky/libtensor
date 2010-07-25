#include <libtensor/libtensor.h>
#include "libtensor_suite.h"

namespace libtensor {

libtensor_suite::libtensor_suite() : libtest::test_suite("libtensor") {

	add_test("abs_index", m_utf_abs_index);
	add_test("addition_schedule", m_utf_addition_schedule);
	add_test("anon_eval", m_utf_anon_eval);
	add_test("bispace", m_utf_bispace);
	add_test("bispace_expr", m_utf_bispace_expr);
	add_test("block_index_space", m_utf_block_index_space);
	add_test("block_index_subspace_builder",
		m_utf_block_index_subspace_builder);
	add_test("block_map", m_utf_block_map);
	add_test("block_tensor", m_utf_block_tensor);
	add_test("btensor", m_utf_btensor);
	add_test("btod_add", m_utf_btod_add);
	add_test("btod_compare", m_utf_btod_compare);
	add_test("btod_contract2", m_utf_btod_contract2);
	add_test("btod_copy", m_utf_btod_copy);
	add_test("btod_diag", m_utf_btod_diag);
	add_test("btod_diagonalize", m_utf_btod_diagonalize);
	add_test("btod_dirsum", m_utf_btod_dirsum);
	add_test("btod_dotprod", m_utf_btod_dotprod);
	add_test("btod_extract", m_utf_btod_extract);
	add_test("btod_import_raw", m_utf_btod_import_raw);
	add_test("btod_mult", m_utf_btod_mult);
	add_test("btod_mult1", m_utf_btod_mult1);
	add_test("btod_print", m_utf_btod_print);
	add_test("btod_random", m_utf_btod_random);
	add_test("btod_read", m_utf_btod_read);
	add_test("btod_scale", m_utf_btod_scale);
	add_test("btod_select", m_utf_btod_select);
	add_test("btod_set", m_utf_btod_set);
	add_test("btod_set_diag", m_utf_btod_set_diag);
	add_test("btod_set_elem", m_utf_btod_set_elem);
	add_test("btod_sum", m_utf_btod_sum);
	add_test("btod_symmetrize", m_utf_btod_symmetrize);
	add_test("btod_trace", m_utf_btod_trace);
	add_test("btod_tridiagonalize", m_utf_btod_tridiagonalize);
	add_test("contract", m_utf_contract);
	add_test("contraction2", m_utf_contraction2);
	add_test("contraction2_list_builder", m_utf_contraction2_list_builder);
	add_test("diag", m_utf_diag);
	add_test("dimensions", m_utf_dimensions);
	add_test("direct_block_tensor", m_utf_direct_block_tensor);
	add_test("direct_btensor", m_utf_direct_btensor);
	add_test("direct_eval", m_utf_direct_eval);
	add_test("direct_product", m_utf_direct_product);
	add_test("dirsum", m_utf_dirsum);
	add_test("dot_product", m_utf_dot_product);
	add_test("expr", m_utf_expr);
	add_test("global_timings", m_utf_global_timings);
	add_test("immutable", m_utf_immutable);
	add_test("index", m_utf_index);
	add_test("index_range", m_utf_index_range);
	add_test("labeled_btensor", m_utf_labeled_btensor);
	add_test("letter", m_utf_letter);
	add_test("letter_expr", m_utf_letter_expr);
	add_test("mask", m_utf_mask);
	add_test("mp_safe_tensor", m_utf_mp_safe_tensor);
	add_test("orbit", m_utf_orbit);
	add_test("orbit_list", m_utf_orbit_list);
	add_test("permutation", m_utf_permutation);
	add_test("permutation_builder", m_utf_permutation_builder);
	add_test("permutation_group", m_utf_permutation_group);
	add_test("point_group_table", m_utf_point_group_table);
	add_test("product_table_container", m_utf_product_table_container);
	add_test("sequence", m_utf_sequence);
	add_test("se_label", m_utf_se_label);
	add_test("se_part", m_utf_se_part);
	add_test("se_perm", m_utf_se_perm);
	add_test("so_add", m_utf_so_add);
	add_test("so_add_impl_label", m_utf_so_add_impl_label);
	add_test("so_add_impl_perm", m_utf_so_add_impl_perm);
	add_test("so_concat", m_utf_so_concat);
	add_test("so_concat_impl_label", m_utf_so_concat_impl_label);
	add_test("so_concat_impl_perm", m_utf_so_concat_impl_perm);
	add_test("so_copy", m_utf_so_copy);
	add_test("so_mult", m_utf_so_mult);
	add_test("so_mult_impl_label", m_utf_so_mult_impl_label);
	add_test("so_mult_impl_perm", m_utf_so_mult_impl_perm);
	add_test("so_permute_impl_label", m_utf_so_permute_impl_label);
	add_test("so_proj_down", m_utf_so_proj_down);
	add_test("so_proj_down_impl_perm", m_utf_so_proj_down_impl_perm);
	add_test("so_proj_up", m_utf_so_proj_up);
	add_test("so_proj_up_impl_label", m_utf_so_proj_up_impl_label);
	add_test("so_proj_up_impl_perm", m_utf_so_proj_up_impl_perm);
	add_test("so_symmetrize", m_utf_so_symmetrize);
	add_test("so_union", m_utf_so_union);
	add_test("so_union_impl_perm", m_utf_so_union_impl_perm);
	add_test("symm", m_utf_symm);
	add_test("symmetry", m_utf_symmetry);
	add_test("symmetry_element_set", m_utf_symmetry_element_set);
	add_test("symmetry_element_set_adapter",
		m_utf_symmetry_element_set_adapter);
	add_test("task_batch", m_utf_task_batch);
	add_test("tensor", m_utf_tensor);
	add_test("timer", m_utf_timer);
	add_test("timings", m_utf_timings);
	add_test("tod_add", m_utf_tod_add);
	add_test("tod_btconv", m_utf_tod_btconv);
	add_test("tod_compare", m_utf_tod_compare);
	add_test("tod_contract2", m_utf_tod_contract2);
	add_test("tod_copy", m_utf_tod_copy);
	add_test("tod_diag", m_utf_tod_diag);
	add_test("tod_dirsum", m_utf_tod_dirsum);
	add_test("tod_dotprod", m_utf_tod_dotprod);
	add_test("tod_extract", m_utf_tod_extract);
	add_test("tod_import_raw", m_utf_tod_import_raw);
	add_test("tod_mult", m_utf_tod_mult);
	add_test("tod_mult1", m_utf_tod_mult1);
	add_test("tod_random", m_utf_tod_random);
	add_test("tod_scale", m_utf_tod_scale);
	add_test("tod_scatter", m_utf_tod_scatter);
	add_test("tod_select", m_utf_tod_select);
	add_test("tod_set", m_utf_tod_set);
	add_test("tod_set_diag", m_utf_tod_set_diag);
	add_test("tod_set_elem", m_utf_tod_set_elem);
	add_test("tod_sum", m_utf_tod_sum);
	add_test("tod_symcontract2", m_utf_tod_symcontract2);
	add_test("tod_trace", m_utf_tod_trace);
	add_test("trace", m_utf_trace);
	add_test("version", m_utf_version);
}

}

