#include <libtensor/libtensor.h>
#include "libtensor_tod_suite.h"

namespace libtensor {

libtensor_tod_suite::libtensor_tod_suite() : libtest::test_suite("libtensor") {

	add_test("contraction2", m_utf_contraction2);
	add_test("contraction2_list_builder", m_utf_contraction2_list_builder);
	add_test("tod_add", m_utf_tod_add);
	add_test("tod_btconv", m_utf_tod_btconv);
	add_test("tod_compare", m_utf_tod_compare);
	add_test("tod_contract2", m_utf_tod_contract2);
	add_test("tod_copy", m_utf_tod_copy);
	add_test("tod_diag", m_utf_tod_diag);
	add_test("tod_dirsum", m_utf_tod_dirsum);
	add_test("tod_dotprod", m_utf_tod_dotprod);
	add_test("tod_ewmult2", m_utf_tod_ewmult2);
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
}

}

