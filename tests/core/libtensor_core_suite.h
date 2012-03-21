#ifndef LIBTENSOR_LIBTENSOR_CORE_SUITE_H
#define LIBTENSOR_LIBTENSOR_CORE_SUITE_H

#include <libtest/test_suite.h>
#include "abs_index_test.h"
#include "block_index_space_test.h"
#include "block_index_space_product_builder_test.h"
#include "block_index_subspace_builder_test.h"
#include "block_map_test.h"
#include "block_tensor_test.h"
#include "dense_tensor_test.h"
#include "dimensions_test.h"
#include "direct_block_tensor_test.h"
#include "immutable_test.h"
#include "index_test.h"
#include "index_range_test.h"
#include "mask_test.h"
#include "mp_safe_tensor_test.h"
#include "orbit_test.h"
#include "orbit_list_test.h"
#include "permutation_test.h"
#include "permutation_builder_test.h"
#include "sequence_test.h"
#include "symmetry_test.h"
#include "symmetry_element_set_test.h"
#include "task_batch_test.h"
#include "transf_list_test.h"
#include "version_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests Tests
    \brief Unit tests of individual classes
    \ingroup libtensor
 **/

/** \defgroup libtensor_tests_core Tests of core components
    \brief Unit tests of the core components of libtensor
    \ingroup libtensor_tests
 **/

/** \brief Test suite for the core components of the tensor library (libtensor)
    \ingroup libtensor_tests

    This suite runs the following tests:
    \li libtensor::abs_index_test
    \li libtensor::block_index_space_test
    \li libtensor::block_index_subspace_builder_test
    \li libtensor::block_map_test
    \li libtensor::block_tensor_test
    \li libtensor::dense_tensor_test
    \li libtensor::dimensions_test
    \li libtensor::direct_block_tensor_test
    \li libtensor::immutable_test
    \li libtensor::index_test
    \li libtensor::index_range_test
    \li libtensor::mask_test
    \li libtensor::mp_safe_tensor_test
    \li libtensor::orbit_test
    \li libtensor::orbit_list_test
    \li libtensor::permutation_test
    \li libtensor::permutation_builder_test
    \li libtensor::sequence_test
    \li libtensor::symmetry_test
    \li libtensor::symmetry_element_set_test
    \li libtensor::task_batch_test
    \li libtensor::transf_list_test
    \li libtensor::version_test
**/
class libtensor_core_suite : public libtest::test_suite {
private:
    unit_test_factory<abs_index_test> m_utf_abs_index;
    unit_test_factory<block_index_space_test> m_utf_block_index_space;
    unit_test_factory<block_index_subspace_builder_test>
		m_utf_block_index_subspace_builder;
    unit_test_factory<block_map_test> m_utf_block_map;
    unit_test_factory<block_tensor_test> m_utf_block_tensor;
    unit_test_factory<dense_tensor_test> m_utf_dense_tensor;
    unit_test_factory<dimensions_test> m_utf_dimensions;
    unit_test_factory<direct_block_tensor_test> m_utf_direct_block_tensor;
    unit_test_factory<immutable_test> m_utf_immutable;
    unit_test_factory<index_test> m_utf_index;
    unit_test_factory<index_range_test> m_utf_index_range;
    unit_test_factory<mask_test> m_utf_mask;
    unit_test_factory<mp_safe_tensor_test> m_utf_mp_safe_tensor;
    unit_test_factory<orbit_test> m_utf_orbit;
    unit_test_factory<orbit_list_test> m_utf_orbit_list;
    unit_test_factory<permutation_test> m_utf_permutation;
    unit_test_factory<permutation_builder_test> m_utf_permutation_builder;
    unit_test_factory<sequence_test> m_utf_sequence;
    unit_test_factory<symmetry_test> m_utf_symmetry;
    unit_test_factory<symmetry_element_set_test> m_utf_symmetry_element_set;
    unit_test_factory<task_batch_test> m_utf_task_batch;
    unit_test_factory<transf_list_test> m_utf_transf_list;
    unit_test_factory<version_test> m_utf_version;

public:
    //!	Creates the suite
    libtensor_core_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CORE_SUITE_H

