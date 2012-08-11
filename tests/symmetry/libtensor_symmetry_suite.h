#ifndef LIBTENSOR_LIBTENSOR_SYMMETRY_SUITE_H
#define LIBTENSOR_LIBTENSOR_SYMMETRY_SUITE_H

#include <libtest/test_suite.h>
#include "block_labeling_test.h"
#include "combine_label_test.h"
#include "combine_part_test.h"
#include "eval_sequence_list_test.h"
#include "er_reduce_test.h"
#include "evaluation_rule_test.h"
#include "permutation_group_test.h"
#include "point_group_table_test.h"
#include "product_rule_test.h"
#include "product_table_container_test.h"
#include "se_label_test.h"
#include "se_part_test.h"
#include "se_perm_test.h"
#include "so_apply_se_label_test.h"
#include "so_apply_se_part_test.h"
#include "so_apply_se_perm_test.h"
#include "so_apply_test.h"
#include "so_dirprod_se_label_test.h"
#include "so_dirprod_se_part_test.h"
#include "so_dirprod_se_perm_test.h"
#include "so_dirprod_test.h"
#include "so_dirsum_se_label_test.h"
#include "so_dirsum_se_part_test.h"
#include "so_dirsum_se_perm_test.h"
#include "so_dirsum_test.h"
#include "so_copy_test.h"
#include "so_merge_se_label_test.h"
#include "so_merge_se_part_test.h"
#include "so_merge_se_perm_test.h"
#include "so_merge_test.h"
#include "so_permute_se_label_test.h"
#include "so_permute_se_part_test.h"
#include "so_permute_se_perm_test.h"
#include "so_reduce_se_label_test.h"
#include "so_reduce_se_part_test.h"
#include "so_reduce_se_perm_test.h"
#include "so_reduce_test.h"
#include "so_symmetrize_se_label_test.h"
#include "so_symmetrize_se_part_test.h"
#include "so_symmetrize_test.h"
#include "symmetry_element_set_adapter_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_sym Tests of symmetry components
    \brief Unit tests of the symmetry components in libtensor
    \ingroup libtensor_tests
 **/

/** \brief Test suite for the symmetry in the tensor library (libtensor)
    \ingroup libtensor_tests

    This suite runs the following tests:
    \li libtensor::block_labeling_test
    \li libtensor::combine_label_test
    \li libtensor::combine_part_test
    \li libtensor::eval_sequence_list_test
    \li libtensor::er_reduce_test
    \li libtensor::evaluation_rule_test
    \li libtensor::partition_set_test
    \li libtensor::permutation_group_test
    \li libtensor::point_group_table_test
    \li libtensor::product_rule_test
    \li libtensor::product_table_container_test
    \li libtensor::se_label_test
    \li libtensor::se_part_test
    \li libtensor::se_perm_test
    \li libtensor::so_apply_se_label_test
    \li libtensor::so_apply_se_part_test
    \li libtensor::so_apply_se_perm_test
    \li libtensor::so_apply_test
    \li libtensor::so_copy_test
    \li libtensor::so_dirprod_se_label_test
    \li libtensor::so_dirprod_se_part_test
    \li libtensor::so_dirprod_se_perm_test
    \li libtensor::so_dirprod_test
    \li libtensor::so_dirsum_se_label_test
    \li libtensor::so_dirsum_se_part_test
    \li libtensor::so_dirsum_se_perm_test
    \li libtensor::so_dirsum_test
    \li libtensor::so_merge_se_label_test
    \li libtensor::so_merge_se_part_test
    \li libtensor::so_merge_se_perm_test
    \li libtensor::so_merge_test
    \li libtensor::so_permute_se_label_test
    \li libtensor::so_permute_se_part_test
    \li libtensor::so_permute_se_perm_test
    \li libtensor::so_reduce_se_label_test
    \li libtensor::so_reduce_se_part_test
    \li libtensor::so_reduce_se_perm_test
    \li libtensor::so_reduce_test
    \li libtensor::so_symmetrize_se_label_test
    \li libtensor::so_symmetrize_test
    \li libtensor::symmetry_element_set_adapter_test

 **/
class libtensor_symmetry_suite : public libtest::test_suite {
private:
    unit_test_factory<block_labeling_test> m_utf_block_labeling;
    unit_test_factory<combine_label_test> m_utf_combine_label;
    unit_test_factory<combine_part_test> m_utf_combine_part;
    unit_test_factory<eval_sequence_list_test> m_utf_eval_sequence_list;
    unit_test_factory<evaluation_rule_test> m_utf_er_reduce;
    unit_test_factory<evaluation_rule_test> m_utf_evaluation_rule;
    unit_test_factory<permutation_group_test> m_utf_permutation_group;
    unit_test_factory<point_group_table_test> m_utf_point_group_table;
    unit_test_factory<product_rule_test> m_utf_product_rule;
    unit_test_factory<product_table_container_test>
        m_utf_product_table_container;
    unit_test_factory<se_label_test> m_utf_se_label;
    unit_test_factory<se_part_test> m_utf_se_part;
    unit_test_factory<se_perm_test> m_utf_se_perm;
    unit_test_factory<so_apply_se_label_test> m_utf_so_apply_se_label;
    unit_test_factory<so_apply_se_part_test> m_utf_so_apply_se_part;
    unit_test_factory<so_apply_se_perm_test> m_utf_so_apply_se_perm;
    unit_test_factory<so_apply_test> m_utf_so_apply;
    unit_test_factory<so_copy_test> m_utf_so_copy;
    unit_test_factory<so_dirprod_se_label_test> m_utf_so_dirprod_se_label;
    unit_test_factory<so_dirprod_se_part_test> m_utf_so_dirprod_se_part;
    unit_test_factory<so_dirprod_se_perm_test> m_utf_so_dirprod_se_perm;
    unit_test_factory<so_dirprod_test> m_utf_so_dirprod;
    unit_test_factory<so_dirsum_se_label_test> m_utf_so_dirsum_se_label;
    unit_test_factory<so_dirsum_se_part_test> m_utf_so_dirsum_se_part;
    unit_test_factory<so_dirsum_se_perm_test> m_utf_so_dirsum_se_perm;
    unit_test_factory<so_dirsum_test> m_utf_so_dirsum;
    unit_test_factory<so_merge_se_label_test> m_utf_so_merge_se_label;
    unit_test_factory<so_merge_se_part_test> m_utf_so_merge_se_part;
    unit_test_factory<so_merge_se_perm_test> m_utf_so_merge_se_perm;
    unit_test_factory<so_merge_test> m_utf_so_merge;
    unit_test_factory<so_permute_se_label_test> m_utf_so_permute_se_label;
    unit_test_factory<so_permute_se_part_test> m_utf_so_permute_se_part;
    unit_test_factory<so_permute_se_perm_test> m_utf_so_permute_se_perm;
    unit_test_factory<so_reduce_se_label_test> m_utf_so_reduce_se_label;
    unit_test_factory<so_reduce_se_part_test> m_utf_so_reduce_se_part;
    unit_test_factory<so_reduce_se_perm_test> m_utf_so_reduce_se_perm;
    unit_test_factory<so_reduce_test> m_utf_so_reduce;
    unit_test_factory<so_symmetrize_se_label_test> m_utf_so_symmetrize_se_label;
    unit_test_factory<so_symmetrize_se_part_test> m_utf_so_symmetrize_se_part;
    unit_test_factory<so_symmetrize_test> m_utf_so_symmetrize;
    unit_test_factory<symmetry_element_set_adapter_test>
        m_utf_symmetry_element_set_adapter;

public:
    //!    Creates the suite
    libtensor_symmetry_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SYMMETRY_SUITE_H

