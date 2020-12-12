#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_set_diag.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include "btod_set_diag_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_set_diag_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();

    } catch (...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void btod_set_diag_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_1()";

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    symmetry<2, double> sym(bis);
    sequence<2, size_t> msk(1);

    test_generic(testname, bis, sym, 0.0);
    test_generic(testname, bis, sym, 11.5);
    test_generic(testname, bis, sym, msk, 0.0);
    test_generic(testname, bis, sym, msk, 11.5);
}


void btod_set_diag_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_2()";

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    sequence<2, size_t> msk(1);

    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 3);

    symmetry<2, double> sym(bis);

    test_generic(testname, bis, sym, 0.0);
    test_generic(testname, bis, sym, 11.6);
    test_generic(testname, bis, sym, msk, 0.0);
    test_generic(testname, bis, sym, msk, 11.6);
}


void btod_set_diag_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_3()";

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    sequence<2, size_t> msk(1);

    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 3);
    bis.split(m, 8);

    symmetry<2, double> sym(bis);

    test_generic(testname, bis, sym, 0.0);
    test_generic(testname, bis, sym, 11.7);
    test_generic(testname, bis, sym, msk, 0.0);
    test_generic(testname, bis, sym, msk, 11.7);
}


void btod_set_diag_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_4()";

    libtensor::index<4> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    sequence<4, size_t> msk(1);

    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 3);
    bis.split(m, 8);

    symmetry<4, double> sym(bis);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> elem1(permutation<4>().permute(0, 1).permute(1, 2).
        permute(2, 3), tr0);
    se_perm<4, double> elem2(permutation<4>().permute(0, 1), tr0);
    sym.insert(elem1);
    sym.insert(elem2);

    test_generic(testname, bis, sym, -2.0);
    test_generic(testname, bis, sym, 0.12);
    test_generic(testname, bis, sym, msk, -2.0);
    test_generic(testname, bis, sym, msk, 0.12);
}


void btod_set_diag_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_5()";

    libtensor::index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    sequence<2, size_t> msk(1);

    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);

    libtensor::index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;

    symmetry<2, double> sym(bis);
    scalar_transf<double> tr0, tr1(-1.);
    se_part<2, double> elem1(bis, m, 2);
    elem1.add_map(i00, i11, tr0);
    elem1.add_map(i01, i10, tr0);
    sym.insert(elem1);

    test_generic(testname, bis, sym, 0.0);
    test_generic(testname, bis, sym, -1.3);
    test_generic(testname, bis, sym, msk, 0.0);
    test_generic(testname, bis, sym, msk, -1.3);
}


void btod_set_diag_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "btod_set_diag_test::test_6()";

    libtensor::index<4> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    sequence<4, size_t> msk(1);
    msk[1] = 2; msk[3] = 2;

    mask<4> m1, m2;
    m1[0] = true; m2[1] = true; m1[2] = true; m2[3] = true;
    bis.split(m1, 3);
    bis.split(m1, 8);
    bis.split(m2, 2);
    bis.split(m2, 7);

    symmetry<4, double> sym(bis);
    scalar_transf<double> tr0;
    se_perm<4, double> elem1(permutation<4>().permute(0, 2).permute(1, 3), tr0);
    sym.insert(elem1);

    test_generic(testname, bis, sym, msk, 0.0);
    test_generic(testname, bis, sym, msk, 0.12);
}


template<size_t N>
void btod_set_diag_test::test_generic(const char *testname,
    const block_index_space<N> &bis, const symmetry<N, double> &sym,
    double d) throw(libtest::test_exception) {

    typedef allocator<double> allocator_t;

    try {

    block_tensor<N, double, allocator_t> bt(bis);
    dense_tensor<N, double, allocator_t> t(bis.get_dims()), t_ref(bis.get_dims());

    //  Fill in random data & make reference

    {
        block_tensor_ctrl<N, double> ctrl(bt);
        so_copy<N, double>(sym).perform(ctrl.req_symmetry());
    }
    btod_random<N>().perform(bt);
    tod_btconv<N>(bt).perform(t_ref);
    tod_set_diag<N>(d).perform(true, t_ref);

    //  Perform the operation

    btod_set_diag<N>(d).perform(bt);
    tod_btconv<N>(bt).perform(t);

    //  Compare against the reference

    compare_ref<N>::compare(testname, t, t_ref, 0.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<size_t N>
void btod_set_diag_test::test_generic(const char *testname,
    const block_index_space<N> &bis, const symmetry<N, double> &sym,
    const sequence<N, size_t> &msk, double d) throw(libtest::test_exception) {

    typedef allocator<double> allocator_t;

    try {

    block_tensor<N, double, allocator_t> bt(bis);
    dense_tensor<N, double, allocator_t> t(bis.get_dims()), t_ref(bis.get_dims());

    //  Fill in random data & make reference

    {
        block_tensor_ctrl<N, double> ctrl(bt);
        so_copy<N, double>(sym).perform(ctrl.req_symmetry());
    }
    btod_random<N>().perform(bt);
    tod_btconv<N>(bt).perform(t_ref);
    tod_set_diag<N>(msk, d).perform(true, t_ref);

    //  Perform the operation

    btod_set_diag<N>(msk, d).perform(bt);
    tod_btconv<N>(bt).perform(t);

    //  Compare against the reference

    compare_ref<N>::compare(testname, t, t_ref, 0.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
