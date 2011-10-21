#ifndef LIBTENSOR_LIBTENSOR_BTOD_SUITE_H
#define LIBTENSOR_LIBTENSOR_BTOD_SUITE_H

#include <libtest/test_suite.h>
#include "addition_schedule_test.h"
#include "btod_add_test.h"
#include "btod_apply_test.h"
#include "btod_cholesky_test.h"
#include "btod_compare_test.h"
#include "btod_contract2_test.h"
#include "btod_copy_test.h"
#include "btod_diag_test.h"
#include "btod_diagonalize_test.h"
#include "btod_dirsum_test.h"
#include "btod_dotprod_test.h"
#include "btod_ewmult2_test.h"
#include "btod_extract_test.h"
#include "btod_import_raw_test.h"
#include "btod_import_raw_stream_test.h"
#include "btod_mult_test.h"
#include "btod_mult1_test.h"
#include "btod_print_test.h"
#include "btod_random_test.h"
#include "btod_read_test.h"
#include "btod_scale_test.h"
#include "btod_select_test.h"
#include "btod_set_test.h"
#include "btod_set_diag_test.h"
#include "btod_set_elem_test.h"
#include "btod_sum_test.h"
#include "btod_symmetrize_test.h"
#include "btod_symmetrize3_test.h"
#include "btod_trace_test.h"
#include "btod_tridiagonalize_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_btod Tests of block tensor operations
 	\brief Unit tests of block %tensor operations.
 	\ingroup libtensor_tests
 **/


/**
	\brief Test suite for the block %tensor operations in libtensor
	\ingroup libtensor_tests

	This suite runs the following tests:
	\li libtensor::addition_schedule_test
	\li libtensor::btod_add_test
	\li libtensor::btod_apply_test
        \li libtensor::btod_cholesky_test
	\li libtensor::btod_compare_test
	\li libtensor::btod_contract2_test
	\li libtensor::btod_copy_test
	\li libtensor::btod_diag_test
	\li libtensor::btod_diagonalize_test
	\li libtensor::btod_dirsum_test
	\li libtensor::btod_dotprod_test
	\li libtensor::btod_ewmult2_test
	\li libtensor::btod_extract_test
	\li libtensor::btod_import_raw_test
    \li libtensor::btod_import_raw_stream_test
	\li libtensor::btod_mult_test
	\li libtensor::btod_mult1_test
	\li libtensor::btod_print_test
	\li libtensor::btod_random_test
	\li libtensor::btod_read_test
	\li libtensor::btod_scale_test
	\li libtensor::btod_select_test
	\li libtensor::btod_set_test
	\li libtensor::btod_set_diag_test
	\li libtensor::btod_set_elem_test
	\li libtensor::btod_sum_test
	\li libtensor::btod_symmetrize_test
	\li libtensor::btod_symmetrize3_test
	\li libtensor::btod_trace_test
	\li libtensor::btod_tridiagonalize_test
**/
class libtensor_btod_suite : public libtest::test_suite {
private:
	unit_test_factory<addition_schedule_test> m_utf_addition_schedule;
	unit_test_factory<btod_add_test> m_utf_btod_add;
	unit_test_factory<btod_apply_test> m_utf_btod_apply;
	unit_test_factory<btod_cholesky_test> m_utf_btod_cholesky;
	unit_test_factory<btod_compare_test> m_utf_btod_compare;
	unit_test_factory<btod_contract2_test> m_utf_btod_contract2;
	unit_test_factory<btod_copy_test> m_utf_btod_copy;
	unit_test_factory<btod_diag_test> m_utf_btod_diag;
	unit_test_factory<btod_diagonalize_test> m_utf_btod_diagonalize;
	unit_test_factory<btod_dirsum_test> m_utf_btod_dirsum;
	unit_test_factory<btod_dotprod_test> m_utf_btod_dotprod;
	unit_test_factory<btod_ewmult2_test> m_utf_btod_ewmult2;
	unit_test_factory<btod_extract_test> m_utf_btod_extract;
	unit_test_factory<btod_import_raw_test> m_utf_btod_import_raw;
	unit_test_factory<btod_import_raw_stream_test> m_utf_btod_import_raw_stream;
	unit_test_factory<btod_mult_test> m_utf_btod_mult;
	unit_test_factory<btod_mult1_test> m_utf_btod_mult1;
	unit_test_factory<btod_print_test> m_utf_btod_print;
	unit_test_factory<btod_random_test> m_utf_btod_random;
	unit_test_factory<btod_read_test> m_utf_btod_read;
	unit_test_factory<btod_scale_test> m_utf_btod_scale;
	unit_test_factory<btod_select_test> m_utf_btod_select;
	unit_test_factory<btod_set_test> m_utf_btod_set;
	unit_test_factory<btod_set_diag_test> m_utf_btod_set_diag;
	unit_test_factory<btod_set_elem_test> m_utf_btod_set_elem;
	unit_test_factory<btod_sum_test> m_utf_btod_sum;
	unit_test_factory<btod_symmetrize_test> m_utf_btod_symmetrize;
	unit_test_factory<btod_symmetrize3_test> m_utf_btod_symmetrize3;
	unit_test_factory<btod_trace_test> m_utf_btod_trace;
	unit_test_factory<btod_tridiagonalize_test> m_utf_btod_tridiagonalize;

public:
	//!	Creates the suite
	libtensor_btod_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_BTOD_SUITE_H

