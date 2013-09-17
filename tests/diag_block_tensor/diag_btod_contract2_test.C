#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/diag_block_tensor/diag_block_tensor.h>
#include <libtensor/diag_block_tensor/diag_btod_contract2.h>
#include <libtensor/diag_block_tensor/diag_btod_random.h>
#include <libtensor/diag_block_tensor/tod_conv_diag_block_tensor.h>
#include "../compare_ref.h"
#include "diag_btod_contract2_test.h"

namespace libtensor {


void diag_btod_contract2_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void diag_btod_contract2_test::test_1() {

    static const char *testname = "diag_btod_contract2_test::test_1()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);

    index<2> i00;

    diag_tensor_subspace<2> ss_00(0);

    //  Set up the structure of A & B

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        {
            diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
            diag_tensor_wr_ctrl<2, double> c00(b00);
            c00.req_add_subspace(ss_00);
            ca.ret_block(i00);
        }
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        {
            diag_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
            diag_tensor_wr_ctrl<2, double> c00(b00);
            c00.req_add_subspace(ss_00);
            cb.ret_block(i00);
        }
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);

    //  Compute C

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);
    diag_btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

    //  Compute the reference

    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);
    tod_conv_diag_block_tensor<2>(btc).perform(tc);
    tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

    //  Compare result with reference

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_contract2_test::test_2() {

    static const char *testname = "diag_btod_contract2_test::test_2()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 8);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis), btc(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tc(dims),
        tc_ref(dims);

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;

    diag_tensor_subspace<2> ss_00(0), ss_11(1);
    ss_11.set_diag_mask(0, m11);

    //  Set up the structure of A & B

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        {
            diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
            diag_tensor_wr_ctrl<2, double> c00(b00);
            c00.req_add_subspace(ss_00);
            ca.ret_block(i00);
        }
        {
            diag_tensor_wr_i<2, double> &b01 = ca.req_block(i01);
            diag_tensor_wr_ctrl<2, double> c01(b01);
            c01.req_add_subspace(ss_00);
            ca.ret_block(i01);
        }
        {
            diag_tensor_wr_i<2, double> &b10 = ca.req_block(i10);
            diag_tensor_wr_ctrl<2, double> c10(b10);
            c10.req_add_subspace(ss_00);
            ca.ret_block(i10);
        }
        {
            diag_tensor_wr_i<2, double> &b11 = ca.req_block(i11);
            diag_tensor_wr_ctrl<2, double> c11(b11);
            c11.req_add_subspace(ss_00);
            ca.ret_block(i11);
        }
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        {
            diag_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
            diag_tensor_wr_ctrl<2, double> c00(b00);
            c00.req_add_subspace(ss_11);
            cb.ret_block(i00);
        }
        {
            diag_tensor_wr_i<2, double> &b01 = cb.req_block(i01);
            diag_tensor_wr_ctrl<2, double> c01(b01);
            c01.req_add_subspace(ss_00);
            cb.ret_block(i01);
        }
        {
            diag_tensor_wr_i<2, double> &b10 = cb.req_block(i10);
            diag_tensor_wr_ctrl<2, double> c10(b10);
            c10.req_add_subspace(ss_00);
            cb.ret_block(i10);
        }
        {
            diag_tensor_wr_i<2, double> &b11 = cb.req_block(i11);
            diag_tensor_wr_ctrl<2, double> c11(b11);
            c11.req_add_subspace(ss_11);
            cb.ret_block(i11);
        }
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);

    //  Compute C

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);
    diag_btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc);

    //  Compute the reference

    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);
    tod_conv_diag_block_tensor<2>(btc).perform(tc);
    tod_contract2<1, 1, 1>(contr, ta, tb).perform(true, tc_ref);

    //  Compare result with reference

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
