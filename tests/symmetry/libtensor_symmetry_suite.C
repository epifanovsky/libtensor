#include <libtensor/libtensor.h>
#include "libtensor_symmetry_suite.h"

namespace libtensor {

libtensor_symmetry_suite::libtensor_symmetry_suite() :
		libtest::test_suite("libtensor") {

	add_test("partition_set", m_utf_partition_set);
	add_test("permutation_group", m_utf_permutation_group);
	add_test("point_group_table", m_utf_point_group_table);
	add_test("product_table_container", m_utf_product_table_container);
	add_test("se_label", m_utf_se_label);
	add_test("se_part", m_utf_se_part);
	add_test("se_perm", m_utf_se_perm);
	add_test("so_add", m_utf_so_add);
	add_test("so_add_impl_label", m_utf_so_add_impl_label);
	add_test("so_add_impl_part", m_utf_so_add_impl_part);
	add_test("so_add_impl_perm", m_utf_so_add_impl_perm);
	add_test("so_apply", m_utf_so_apply);
	add_test("so_apply_impl_label", m_utf_so_apply_impl_label);
	add_test("so_apply_impl_part", m_utf_so_apply_impl_part);
	add_test("so_apply_impl_perm", m_utf_so_apply_impl_perm);
	add_test("so_concat", m_utf_so_concat);
	add_test("so_concat_impl_label", m_utf_so_concat_impl_label);
	add_test("so_concat_impl_part", m_utf_so_concat_impl_part);
	add_test("so_concat_impl_perm", m_utf_so_concat_impl_perm);
	add_test("so_copy", m_utf_so_copy);
	add_test("so_merge", m_utf_so_merge);
	add_test("so_merge_impl_label", m_utf_so_merge_impl_label);
	add_test("so_merge_impl_part", m_utf_so_merge_impl_part);
	add_test("so_merge_impl_perm", m_utf_so_merge_impl_perm);
	add_test("so_mult", m_utf_so_mult);
	add_test("so_mult_impl_label", m_utf_so_mult_impl_label);
	add_test("so_mult_impl_part", m_utf_so_mult_impl_part);
	add_test("so_mult_impl_perm", m_utf_so_mult_impl_perm);
	add_test("so_permute_impl_label", m_utf_so_permute_impl_label);
	add_test("so_permute_impl_part", m_utf_so_permute_impl_part);
	add_test("so_permute_impl_perm", m_utf_so_permute_impl_perm);
	add_test("so_proj_down", m_utf_so_proj_down);
	add_test("so_proj_down_impl_label", m_utf_so_proj_down_impl_label);
	add_test("so_proj_down_impl_part", m_utf_so_proj_down_impl_part);
	add_test("so_proj_down_impl_perm", m_utf_so_proj_down_impl_perm);
	add_test("so_proj_up", m_utf_so_proj_up);
	add_test("so_proj_up_impl_label", m_utf_so_proj_up_impl_label);
	add_test("so_proj_up_impl_part", m_utf_so_proj_up_impl_part);
	add_test("so_proj_up_impl_perm", m_utf_so_proj_up_impl_perm);
	add_test("so_stabilize", m_utf_so_stabilize);
	add_test("so_stabilize_impl_label", m_utf_so_stabilize_impl_label);
	add_test("so_stabilize_impl_part", m_utf_so_stabilize_impl_part);
	add_test("so_stabilize_impl_perm", m_utf_so_stabilize_impl_perm);
	add_test("so_symmetrize", m_utf_so_symmetrize);
	add_test("symmetry_element_set_adapter",
		m_utf_symmetry_element_set_adapter);
}

}

