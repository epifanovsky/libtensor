#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_contract2.h>
#include <libtensor/dense_tensor/tod_scale.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/dense_tensor/tod_set_diag.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_set.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_set_test.h"

namespace libtensor {


void diag_tod_set_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

    test_1();
    test_2();
    test_3();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


void diag_tod_set_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "diag_tod_set_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b, i2c, i2d;
        i2b[0] = 5; i2b[1] = 5;
        i2d[0] = 6; i2d[1] = 6;
        dimensions<2> dims55(index_range<2>(i2a, i2b));
        dimensions<2> dims66(index_range<2>(i2c, i2d));

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 6; i4b[2] = 5; i4b[3] = 6;
        dimensions<4> dims5656(index_range<4>(i4a, i4b));

        mask<4> m0101, m1010;
        m0101[1] = true; m0101[3] = true;
        m1010[0] = true; m1010[2] = true;

        diag_tensor_subspace<4> dts1(1), dts2(1);
        dts1.set_diag_mask(0, m0101);
        dts2.set_diag_mask(0, m1010);

        diag_tensor_space<4> dts(dims5656);
        size_t ssn1 = dts.add_subspace(dts1);
        size_t ssn2 = dts.add_subspace(dts2);
        size_t sz1 = dts.get_subspace_size(ssn1);
        size_t sz2 = dts.get_subspace_size(ssn2);

        dense_tensor<2, double, allocator_t> t55(dims55), t66(dims66),
            t55d(dims55), t66d(dims66);
        tod_set<2>().perform(t55);
        tod_set<2>().perform(t66);
        tod_set<2>().perform(t55d);
        tod_set<2>().perform(t66d);
        tod_set_diag<2>(1.0).perform(t55d);
        tod_set_diag<2>(1.0).perform(t66d);

        diag_tensor<4, double, allocator_t> dt(dts);
        dense_tensor<4, double, allocator_t> t(dims5656), t_ref(dims5656);

        permutation<4> p0213, p3120;
        p0213.permute(1, 2);
        p3120.permute(0, 3);
        contraction2<2, 2, 0> contr1(p0213), contr2(p3120);
        tod_contract2<2, 2, 0>(contr1, t55d, t66).
            perform(true, t_ref);
        tod_contract2<2, 2, 0>(contr2, t66d, t55).
            perform(false, t_ref);

        diag_tod_set<4>().perform(dt);

        tod_conv_diag_tensor<4>(dt).perform(t);
	compare_ref<4>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tod_set_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "diag_tod_set_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b, i2c, i2d;
        i2b[0] = 5; i2b[1] = 5;
        i2d[0] = 6; i2d[1] = 6;
        dimensions<2> dims55(index_range<2>(i2a, i2b));
        dimensions<2> dims66(index_range<2>(i2c, i2d));

        index<4> i4a, i4b;
        i4b[0] = 5; i4b[1] = 6; i4b[2] = 5; i4b[3] = 6;
        dimensions<4> dims5656(index_range<4>(i4a, i4b));

        mask<4> m0101, m1010;
        m0101[1] = true; m0101[3] = true;
        m1010[0] = true; m1010[2] = true;

        diag_tensor_subspace<4> dts1(1), dts2(1);
        dts1.set_diag_mask(0, m0101);
        dts2.set_diag_mask(0, m1010);

        diag_tensor_space<4> dts(dims5656);
        size_t ssn1 = dts.add_subspace(dts1);
        size_t ssn2 = dts.add_subspace(dts2);
        size_t sz1 = dts.get_subspace_size(ssn1);
        size_t sz2 = dts.get_subspace_size(ssn2);

        dense_tensor<2, double, allocator_t> t55(dims55), t66(dims66),
            t55d(dims55), t66d(dims66);
        tod_set<2>(1.0).perform(t55);
        tod_set<2>(1.0).perform(t66);
        tod_set<2>().perform(t55d);
        tod_set<2>().perform(t66d);
        tod_set_diag<2>(1.0).perform(t55d);
        tod_set_diag<2>(1.0).perform(t66d);

        diag_tensor<4, double, allocator_t> dt(dts);
        dense_tensor<4, double, allocator_t> t(dims5656), t_ref(dims5656);

        permutation<4> p0213, p3120;
        p0213.permute(1, 2);
        p3120.permute(0, 3);
        contraction2<2, 2, 0> contr1(p0213), contr2(p3120);
        tod_contract2<2, 2, 0>(contr1, t55d, t66).perform(true, t_ref);
        tod_contract2<2, 2, 0>(contr2, t66d, t55).perform(false, t_ref);

        diag_tod_set<4>(1.0).perform(dt);

        tod_conv_diag_tensor<4>(dt).perform(t);
	compare_ref<4>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_tod_set_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "diag_tod_set_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i2a, i2b;
        i2b[0] = 6; i2b[1] = 6;
        dimensions<2> dims66(index_range<2>(i2a, i2b));

        index<4> i4a, i4b;
        i4b[0] = 6; i4b[1] = 6; i4b[2] = 6; i4b[3] = 6;
        dimensions<4> dims6666(index_range<4>(i4a, i4b));

        mask<4> m0101, m1010;
        m0101[1] = true; m0101[3] = true;
        m1010[0] = true; m1010[2] = true;

        diag_tensor_subspace<4> dts1(2);
        dts1.set_diag_mask(0, m0101);
        dts1.set_diag_mask(1, m1010);

        diag_tensor_space<4> dts(dims6666);
        size_t ssn1 = dts.add_subspace(dts1);
        size_t sz1 = dts.get_subspace_size(ssn1);

        dense_tensor<2, double, allocator_t> t66(dims66), t66d(dims66);
        tod_set<2>(1.0).perform(t66);
        tod_set<2>().perform(t66d);
        tod_set_diag<2>(1.0).perform(t66d);

        diag_tensor<4, double, allocator_t> dt(dts);
        dense_tensor<4, double, allocator_t> t(dims6666), t_ref(dims6666);

        permutation<4> p0213, p3120;
        p0213.permute(1, 2);
        p3120.permute(0, 3);
        contraction2<2, 2, 0> contr1(p0213), contr2(p3120);
        tod_contract2<2, 2, 0>(contr1, t66d, t66d).perform(true, t_ref);
        tod_scale<4>(-2.5).perform(t_ref);

        diag_tod_set<4>(-2.5).perform(dt);

        tod_conv_diag_tensor<4>(dt).perform(t);
	compare_ref<4>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

