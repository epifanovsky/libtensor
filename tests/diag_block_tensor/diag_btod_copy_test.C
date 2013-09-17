#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/diag_block_tensor/diag_block_tensor.h>
#include <libtensor/diag_block_tensor/diag_btod_copy.h>
#include <libtensor/diag_block_tensor/diag_btod_random.h>
#include <libtensor/diag_block_tensor/tod_conv_diag_block_tensor.h>
#include "../compare_ref.h"
#include "diag_btod_copy_test.h"

namespace libtensor {


void diag_btod_copy_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_copy_nosym_1();
        test_copy_nosym_2();
        test_copy_nosym_3();
        test_copy_nosym_4();
        test_add_nosym_1();
        test_add_nosym_2a();
        test_add_nosym_2b();
        test_add_nosym_3();
        test_add_nosym_4();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void diag_btod_copy_test::test_copy_nosym_1() {

    static const char *testname = "diag_btod_copy_test::test_copy_nosym_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> tb(dims), tb_ref(dims);

    tod_conv_diag_block_tensor<2>(bta).perform(tb_ref);

    diag_btod_copy<2>(bta).perform(btb);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_copy_nosym_2() {

    static const char *testname = "diag_btod_copy_test::test_copy_nosym_2()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_tensor_subspace<2> dtss1(1);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
        }
        ca.ret_block(i00);
    }

    diag_btod_random<2>().perform(bta);
    tod_conv_diag_block_tensor<2>(bta).perform(tb_ref);

    diag_btod_copy<2>(bta).perform(btb);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_copy_nosym_3() {

    static const char *testname = "diag_btod_copy_test::test_copy_nosym_3()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_tensor_subspace<2> dtss1(0), dtss2(1);
    dtss2.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        ca.ret_block(i00);
    }

    diag_btod_random<2>().perform(bta);
    tod_conv_diag_block_tensor<2>(bta).perform(tb_ref);

    diag_btod_copy<2>(bta).perform(btb);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_copy_nosym_4() {

    static const char *testname = "diag_btod_copy_test::test_copy_nosym_4()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 8);

    diag_tensor_subspace<2> dtss1(0), dtss2(1);
    dtss2.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        diag_tensor_wr_i<2, double> &b01 = ca.req_block(i01);
        {
            diag_tensor_wr_ctrl<2, double> ct(b01);
            ct.req_add_subspace(dtss2);
        }
        ca.ret_block(i01);
        diag_tensor_wr_i<2, double> &b11 = ca.req_block(i11);
        {
            diag_tensor_wr_ctrl<2, double> ct(b11);
            ct.req_add_subspace(dtss1);
        }
        ca.ret_block(i11);
    }

    diag_btod_random<2>().perform(bta);
    tod_conv_diag_block_tensor<2>(bta).perform(tb_ref);

    diag_btod_copy<2>(bta).perform(btb);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_add_nosym_1() {

    static const char *testname = "diag_btod_copy_test::test_add_nosym_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb_ref);
    tod_copy<2>(ta, -2.0).perform(false, tb_ref);

    diag_btod_copy<2>(bta).perform(btb, -2.0);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_add_nosym_2a() {

    static const char *testname = "diag_btod_copy_test::test_add_nosym_2a()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_tensor_subspace<2> dtss1(1), dtss2(0);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
        }
        ca.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        diag_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss2);
        }
        cb.ret_block(i00);
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);
    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb_ref);
    tod_copy<2>(ta, 0.5).perform(false, tb_ref);

    diag_btod_copy<2>(bta).perform(btb, 0.5);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_add_nosym_2b() {

    static const char *testname = "diag_btod_copy_test::test_add_nosym_2b()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    permutation<2> p10;
    p10.permute(0, 1);

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_tensor_subspace<2> dtss1(1), dtss2(0);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
        }
        ca.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        diag_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss2);
        }
        cb.ret_block(i00);
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);
    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb_ref);
    tod_copy<2>(ta, p10, -0.5).perform(false, tb_ref);

    diag_btod_copy<2>(bta, p10, -0.5).perform(btb, 1.0);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_add_nosym_3() {

    static const char *testname = "diag_btod_copy_test::test_add_nosym_3()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_tensor_subspace<2> dtss1(0), dtss2(1);
    dtss2.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        ca.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        diag_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss2);
        }
        cb.ret_block(i00);
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);
    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb_ref);
    tod_copy<2>(ta, 1.2).perform(false, tb_ref);

    diag_btod_copy<2>(bta).perform(btb, 1.2);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void diag_btod_copy_test::test_add_nosym_4() {

    static const char *testname = "diag_btod_copy_test::test_add_nosym_4()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    permutation<2> p10;
    p10.permute(0, 1);

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, 8);

    diag_tensor_subspace<2> dtss1(0), dtss2(1);
    dtss2.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> ca(bta);
        diag_tensor_wr_i<2, double> &b00 = ca.req_block(i00);
        {
            diag_tensor_wr_ctrl<2, double> ct(b00);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        diag_tensor_wr_i<2, double> &b01 = ca.req_block(i01);
        {
            diag_tensor_wr_ctrl<2, double> ct(b01);
            ct.req_add_subspace(dtss2);
        }
        ca.ret_block(i01);
        diag_tensor_wr_i<2, double> &b11 = ca.req_block(i11);
        {
            diag_tensor_wr_ctrl<2, double> ct(b11);
            ct.req_add_subspace(dtss1);
        }
        ca.ret_block(i11);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> cb(btb);
        diag_tensor_wr_i<2, double> &b01 = cb.req_block(i01);
        {
            diag_tensor_wr_ctrl<2, double> ct(b01);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        cb.ret_block(i01);
        diag_tensor_wr_i<2, double> &b11 = cb.req_block(i11);
        {
            diag_tensor_wr_ctrl<2, double> ct(b11);
            ct.req_add_subspace(dtss1);
            ct.req_add_subspace(dtss2);
        }
        cb.ret_block(i11);
    }

    diag_btod_random<2>().perform(bta);
    diag_btod_random<2>().perform(btb);
    tod_conv_diag_block_tensor<2>(bta).perform(ta);
    tod_conv_diag_block_tensor<2>(btb).perform(tb_ref);
    tod_copy<2>(ta, p10).perform(false, tb_ref);

    diag_btod_copy<2>(bta, p10).perform(btb, 1.0);
    tod_conv_diag_block_tensor<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
