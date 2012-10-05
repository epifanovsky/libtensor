#include <vector>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/diag_block_tensor/diag_block_tensor.h>
#include <libtensor/diag_block_tensor/tod_conv_diag_block_tensor.h>
#include "../compare_ref.h"
#include "tod_conv_diag_block_tensor_test.h"

namespace libtensor {


void tod_conv_diag_block_tensor_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


void tod_conv_diag_block_tensor_test::test_1() {

    static const char *testname = "tod_conv_diag_block_tensor_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 15; i2[1] = 15;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    diag_block_tensor<2, double, allocator_t> bt(bis);
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    tod_conv_diag_block_tensor<2>(bt).perform(t);
    tod_set<2>().perform(t_ref);

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void tod_conv_diag_block_tensor_test::test_2() {

    static const char *testname = "tod_conv_diag_block_tensor_test::test_2()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    size_t ni = 16;

    index<2> i00;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = ni - 1; i2[1] = ni - 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    std::vector<double> v(ni, 0.0);
    for(size_t i = 0; i < ni; i++) v[i] = drand48();

    diag_tensor_subspace<2> dtss1(1);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bt(bis);
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i00);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i00);
    }

    tod_conv_diag_block_tensor<2>(bt).perform(t);

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        double *p = ctrl.req_dataptr();
        for(size_t i = 0; i < ni * ni; i++) p[i] = 0.0;
        for(size_t i = 0; i < ni; i++) p[i * ni + i] = v[i];
        ctrl.ret_dataptr(p);
    }

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void tod_conv_diag_block_tensor_test::test_3() {

    static const char *testname = "tod_conv_diag_block_tensor_test::test_3()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    size_t ni = 8;

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 2 * ni - 1; i2[1] = 2 * ni - 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, ni);

    std::vector<double> v1(ni, 0.0), v2(ni, 0.0), v3(ni, 0.0), v4(ni, 0.0);
    for(size_t i = 0; i < ni; i++) {
        v1[i] = drand48();
        v2[i] = drand48();
        v3[i] = drand48();
        v4[i] = drand48();
    }

    diag_tensor_subspace<2> dtss1(1);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bt(bis);
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i00);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v1[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i01);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v2[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i01);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i10);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v3[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i10);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i11);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v4[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i11);
    }

    tod_conv_diag_block_tensor<2>(bt).perform(t);

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        double *p = ctrl.req_dataptr();
        size_t ni2 = ni * 2;
        for(size_t i = 0; i < ni2 * ni2; i++) p[i] = 0.0;
        for(size_t i = 0; i < ni; i++) p[i * ni2 + i] = v1[i];
        for(size_t i = 0; i < ni; i++) p[i * ni2 + ni + i] = v2[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + i] = v3[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + ni + i] = v4[i];
        ctrl.ret_dataptr(p);
    }

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void tod_conv_diag_block_tensor_test::test_4() {

    static const char *testname = "tod_conv_diag_block_tensor_test::test_4()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    size_t ni = 8;

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 2 * ni - 1; i2[1] = 2 * ni - 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, ni);

    std::vector<double> v1(ni, 0.0), v2(ni, 0.0), v3(ni, 0.0);
    for(size_t i = 0; i < ni; i++) {
        v1[i] = drand48();
        v2[i] = drand48();
        v3[i] = drand48();
    }

    diag_tensor_subspace<2> dtss1(1);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bt(bis);
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        se_perm<2, double> se1(permutation<2>().permute(0, 1),
            scalar_transf<double>());
        btctrl.req_symmetry().insert(se1);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i00);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v1[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i01);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v2[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i01);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i11);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v3[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i11);
    }

    tod_conv_diag_block_tensor<2>(bt).perform(t);

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        double *p = ctrl.req_dataptr();
        size_t ni2 = ni * 2;
        for(size_t i = 0; i < ni2 * ni2; i++) p[i] = 0.0;
        for(size_t i = 0; i < ni; i++) p[i * ni2 + i] = v1[i];
        for(size_t i = 0; i < ni; i++) p[i * ni2 + ni + i] = v2[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + i] = v2[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + ni + i] = v3[i];
        ctrl.ret_dataptr(p);
    }

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void tod_conv_diag_block_tensor_test::test_5() {

    static const char *testname = "tod_conv_diag_block_tensor_test::test_5()";

    typedef std_allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;

    try {

    size_t ni = 8;

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i0, i1, i2;
    i2[0] = 2 * ni - 1; i2[1] = 2 * ni - 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(m11, ni);

    std::vector<double> v1(ni, 0.0), v2(ni, 0.0), v3(ni, 0.0);
    for(size_t i = 0; i < ni; i++) {
        v1[i] = drand48();
        v2[i] = drand48();
        v3[i] = drand48();
    }

    diag_tensor_subspace<2> dtss1(1);
    dtss1.set_diag_mask(0, m11);

    diag_block_tensor<2, double, allocator_t> bt(bis);
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        se_perm<2, double> se1(permutation<2>().permute(0, 1),
            scalar_transf<double>(-1.0));
        btctrl.req_symmetry().insert(se1);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i00);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v1[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i00);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i01);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v2[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i01);
    }
    {
        gen_block_tensor_ctrl<2, bti_traits> btctrl(bt);
        diag_tensor_wr_i<2, double> &blk = btctrl.req_block(i11);
        diag_tensor_wr_ctrl<2, double> tctrl(blk);
        size_t ssn1 = tctrl.req_add_subspace(dtss1);
        double *p = tctrl.req_dataptr(ssn1);
        for(size_t i = 0; i < ni; i++) p[i] = v3[i];
        tctrl.ret_dataptr(ssn1, p);
        btctrl.ret_block(i11);
    }

    tod_conv_diag_block_tensor<2>(bt).perform(t);

    {
        dense_tensor_ctrl<2, double> ctrl(t_ref);
        double *p = ctrl.req_dataptr();
        size_t ni2 = ni * 2;
        for(size_t i = 0; i < ni2 * ni2; i++) p[i] = 0.0;
        for(size_t i = 0; i < ni; i++) p[i * ni2 + i] = v1[i];
        for(size_t i = 0; i < ni; i++) p[i * ni2 + ni + i] = v2[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + i] = -v2[i];
        for(size_t i = 0; i < ni; i++) p[(ni + i) * ni2 + ni + i] = v3[i];
        ctrl.ret_dataptr(p);
    }

    compare_ref<2>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
