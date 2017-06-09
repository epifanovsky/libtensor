#include <libtensor/libtensor.h>
#include "libtensor_dense_tensor_suite.h"

namespace libtensor {

libtensor_dense_tensor_suite::libtensor_dense_tensor_suite() :
    libtest::test_suite("libtensor_dense_tensor") {

    add_test("dense_tensor", m_utf_dense_tensor);
    add_test("tod_add", m_utf_tod_add);
    add_test("tod_apply", m_utf_tod_apply);
    add_test("tod_btconv", m_utf_tod_btconv);
    add_test("tod_compare", m_utf_tod_compare);
    add_test("to_contract2_dims", m_utf_to_contract2_dims);
    add_test("tod_contract2", m_utf_tod_contract2);
    add_test("tod_copy", m_utf_tod_copy);
    add_test("tod_copy_wnd", m_utf_tod_copy_wnd);
    add_test("tod_diag", m_utf_tod_diag);
    add_test("tod_dirsum", m_utf_tod_dirsum);
    add_test("tod_dotprod", m_utf_tod_dotprod);
    add_test("tod_ewmult2", m_utf_tod_ewmult2);
    add_test("tod_extract", m_utf_tod_extract);
    add_test("tod_import_raw", m_utf_tod_import_raw);
    add_test("tod_import_raw_stream", m_utf_tod_import_raw_stream);
    add_test("tod_mult", m_utf_tod_mult);
    add_test("tod_mult1", m_utf_tod_mult1);
    add_test("tod_random", m_utf_tod_random);
    add_test("tod_scale", m_utf_tod_scale);
    add_test("tod_scatter", m_utf_tod_scatter);
    add_test("tod_select", m_utf_tod_select);
    add_test("tod_set", m_utf_tod_set);
    add_test("tof_set", m_utf_tof_set);
    add_test("to_set", m_utf_to_set);
    add_test("tod_set_diag", m_utf_tod_set_diag);
    add_test("tod_set_elem", m_utf_tod_set_elem);
    add_test("tod_size", m_utf_tod_size);
    add_test("tod_trace", m_utf_tod_trace);
    add_test("tod_vmpriority", m_utf_tod_vmpriority);
}

} // namespace libtensor
