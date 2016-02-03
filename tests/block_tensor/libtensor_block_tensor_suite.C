#include "libtensor_block_tensor_suite.h"

namespace libtensor {


libtensor_block_tensor_suite::libtensor_block_tensor_suite() :
    libtest::test_suite("libtensor_block_tensor") {

    add_test("block_tensor", m_utf_block_tensor);
    add_test("direct_block_tensor", m_utf_direct_block_tensor);
    add_test("addition_schedule", m_utf_addition_schedule);
    add_test("bto_contract2_bis", m_utf_bto_contract2_bis);
    add_test("bto_contract2_sym", m_utf_bto_contract2_sym);
    add_test("btod_add", m_utf_btod_add);
    add_test("btod_apply", m_utf_btod_apply);
    add_test("btod_compare", m_utf_btod_compare);
    add_test("btod_contract2", m_utf_btod_contract2);
#ifdef USE_LIBXM
    add_test("btod_contract2_xm", m_utf_btod_contract2_xm);
#endif
    add_test("btod_contract3", m_utf_btod_contract3);
    add_test("btod_copy", m_utf_btod_copy);
    add_test("btod_diag", m_utf_btod_diag);
//    add_test("btod_diagonalize", m_utf_btod_diagonalize);
    add_test("btod_dirsum", m_utf_btod_dirsum);
    add_test("btod_dotprod", m_utf_btod_dotprod);
    add_test("btod_ewmult2", m_utf_btod_ewmult2);
    add_test("btod_extract", m_utf_btod_extract);
    add_test("btod_import_raw", m_utf_btod_import_raw);
    add_test("btod_import_raw_stream", m_utf_btod_import_raw_stream);
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
    add_test("btod_shift_diag", m_utf_btod_shift_diag);
    add_test("btod_sum", m_utf_btod_sum);
    add_test("btod_symcontract3", m_utf_btod_symcontract3);
    add_test("btod_symmetrize2", m_utf_btod_symmetrize2);
    add_test("btod_symmetrize3", m_utf_btod_symmetrize3);
//    add_test("btod_symmetrize4", m_utf_btod_symmetrize4);
    add_test("btod_trace", m_utf_btod_trace);
//    add_test("btod_tridiagonalize", m_utf_btod_tridiagonalize);
    add_test("btod_vmpriority", m_utf_btod_vmpriority);
    add_test("gen_bto_aux_add", m_utf_gen_bto_aux_add);
    add_test("gen_bto_aux_copy", m_utf_gen_bto_aux_copy);
    add_test("gen_bto_contract2_clst_builder",
        m_utf_gen_bto_contract2_clst_builder);
    add_test("gen_bto_dirsum_sym", m_utf_gen_bto_dirsum_sym);
    add_test("gen_bto_symcontract2_sym", m_utf_gen_bto_symcontract2_sym);
    add_test("gen_bto_unfold_symmetry", m_utf_gen_bto_unfold_symmetry);
}


} // namespace libtensor

