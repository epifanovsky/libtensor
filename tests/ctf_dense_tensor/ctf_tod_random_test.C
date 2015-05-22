#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_random.h>
#include "../compare_ref.h"
#include "ctf_tod_random_test.h"

namespace libtensor {


void ctf_tod_random_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();
        test_6();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_random_test::test_1() {

    static const char testname[] = "ctf_tod_random_test::test_1()";

    typedef std_allocator<double> allocator_t;

    index<3> i3a, i3b;
    i3b[0] = 29; i3b[1] = 19; i3b[2] = 40;
    dimensions<3> dims(index_range<3>(i3a, i3b));

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims);
    ctf_dense_tensor<3, double> dta(dims), dtb(dims);

    tod_set<3>(100.0).perform(true, ta);
    tod_set<3>(100.0).perform(true, tb);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_random<3>().perform(true, dta);
    ctf_tod_random<3>().perform(true, dtb);

    ctf_tod_collect<3>(dta).perform(ta);
    ctf_tod_collect<3>(dtb).perform(tb);

    dense_tensor_rd_ctrl<3, double> ca(ta), cb(tb);
    const double *pa = ca.req_const_dataptr(), *pb = cb.req_const_dataptr();
    size_t sz = dims.get_size();

    for(size_t i = 0; i < sz; i++) if(pa[i] < 0.0 || pa[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (A) is out of range.");
    }

    for(size_t i = 0; i < sz; i++) if(pb[i] < 0.0 || pb[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (B) is out of range.");
    }

    size_t ndup = 0;
    for(size_t i = 0; i < sz; i++) if(fabs(pa[i] - pb[i]) < 1e-14) ndup++;
    if(ndup > 0) {
        fail_test(testname, __FILE__, __LINE__, "Too many values are similar.");
    }
}


void ctf_tod_random_test::test_2() {

    static const char testname[] = "ctf_tod_random_test::test_2()";

    typedef std_allocator<double> allocator_t;

    index<3> i3a, i3b;
    i3b[0] = 29; i3b[1] = 19; i3b[2] = 40;
    dimensions<3> dims(index_range<3>(i3a, i3b));

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims);
    ctf_dense_tensor<3, double> dta(dims), dtb(dims);

    tod_set<3>(-1.0).perform(true, ta);
    tod_set<3>(-1.0).perform(true, tb);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_random<3>(2.0).perform(false, dta);
    ctf_tod_random<3>(2.0).perform(false, dtb);

    ctf_tod_collect<3>(dta).perform(ta);
    ctf_tod_collect<3>(dtb).perform(tb);

    dense_tensor_rd_ctrl<3, double> ca(ta), cb(tb);
    const double *pa = ca.req_const_dataptr(), *pb = cb.req_const_dataptr();
    size_t sz = dims.get_size();

    for(size_t i = 0; i < sz; i++) if(pa[i] < -1.0 || pa[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (A) is out of range.");
    }

    for(size_t i = 0; i < sz; i++) if(pb[i] < -1.0 || pb[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (B) is out of range.");
    }

    size_t ndup = 0;
    for(size_t i = 0; i < sz; i++) if(fabs(pa[i] - pb[i]) < 1e-14) ndup++;
    if(ndup > 0) {
        fail_test(testname, __FILE__, __LINE__, "Too many values are similar.");
    }
}


void ctf_tod_random_test::test_3() {

    static const char testname[] = "ctf_tod_random_test::test_3()";

    typedef std_allocator<double> allocator_t;

    index<1> i1a, i1b;
    i1b[0] = 2;
    dimensions<1> dims(index_range<1>(i1a, i1b));

    dense_tensor<1, double, allocator_t> ta(dims);
    ctf_dense_tensor<1, double> dta(dims);

    tod_set<1>(100.0).perform(true, ta);
    ctf_tod_distribute<1>(ta).perform(dta);
    ctf_tod_random<1>().perform(true, dta);
    ctf_tod_collect<1>(dta).perform(ta);

    dense_tensor_rd_ctrl<1, double> ca(ta);
    const double *pa = ca.req_const_dataptr();
    size_t sz = dims.get_size();

    for(size_t i = 0; i < sz; i++) if(pa[i] < 0.0 || pa[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (A) is out of range.");
    }
}


void ctf_tod_random_test::test_4() {

    static const char testname[] = "ctf_tod_random_test::test_4()";

    typedef std_allocator<double> allocator_t;

    index<2> i2a, i2b;
    i2b[0] = 19; i2b[1] = 19;
    dimensions<2> dims(index_range<2>(i2a, i2b));
    sequence<2, unsigned> grp(0), gind(0);
    ctf_symmetry<2, double> sym(grp, gind);

    dense_tensor<2, double, allocator_t> ta(dims), tb(dims);
    ctf_dense_tensor<2, double> dta(dims, sym), dtb(dims, sym);

    tod_set<2>(100.0).perform(true, ta);
    tod_set<2>(100.0).perform(true, tb);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);

    ctf_tod_random<2>().perform(true, dta);
    ctf_tod_random<2>().perform(true, dtb);

    ctf_tod_collect<2>(dta).perform(ta);
    ctf_tod_collect<2>(dtb).perform(tb);

    dense_tensor_rd_ctrl<2, double> ca(ta), cb(tb);
    const double *pa = ca.req_const_dataptr(), *pb = cb.req_const_dataptr();
    size_t sz = dims.get_size();

    for(size_t i = 0; i < sz; i++) if(pa[i] < 0.0 || pa[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (A) is out of range.");
    }

    for(size_t i = 0; i < sz; i++) if(pb[i] < 0.0 || pb[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (B) is out of range.");
    }

    size_t ndup = 0;
    for(size_t i = 0; i < sz; i++) if(fabs(pa[i] - pb[i]) < 1e-14) ndup++;
    if(ndup > 0) {
        fail_test(testname, __FILE__, __LINE__, "Too many values are similar.");
    }
}


void ctf_tod_random_test::test_5() {

    static const char testname[] = "ctf_tod_random_test::test_5()";

    typedef std_allocator<double> allocator_t;

    index<3> i3a, i3b;
    i3b[0] = 19; i3b[1] = 19; i3b[2] = 19;
    dimensions<3> dims(index_range<3>(i3a, i3b));
    sequence<3, unsigned> grp(0), gind(0);
    ctf_symmetry<3, double> sym(grp, gind);

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims);
    ctf_dense_tensor<3, double> dta(dims, sym), dtb(dims, sym);

    tod_set<3>(100.0).perform(true, ta);
    tod_set<3>(100.0).perform(true, tb);

    ctf_tod_distribute<3>(ta).perform(dta);
    ctf_tod_distribute<3>(tb).perform(dtb);

    ctf_tod_random<3>().perform(true, dta);
    ctf_tod_random<3>().perform(true, dtb);

    ctf_tod_collect<3>(dta).perform(ta);
    ctf_tod_collect<3>(dtb).perform(tb);

    dense_tensor_rd_ctrl<3, double> ca(ta), cb(tb);
    const double *pa = ca.req_const_dataptr(), *pb = cb.req_const_dataptr();
    size_t sz = dims.get_size();

    for(size_t i = 0; i < sz; i++) if(pa[i] < 0.0 || pa[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (A) is out of range.");
    }

    for(size_t i = 0; i < sz; i++) if(pb[i] < 0.0 || pb[i] > 1.0) {
        fail_test(testname, __FILE__, __LINE__,
            "Random value (B) is out of range.");
    }

    size_t ndup = 0;
    for(size_t i = 0; i < sz; i++) if(fabs(pa[i] - pb[i]) < 1e-14) ndup++;
    if(ndup > 0) {
        fail_test(testname, __FILE__, __LINE__, "Too many values are similar.");
    }
}


void ctf_tod_random_test::test_6() {

    static const char testname[] = "ctf_tod_random_test::test_6()";

    typedef std_allocator<double> allocator_t;

    try {

    index<3> i3a, i3b;
    i3b[0] = 19; i3b[1] = 19; i3b[2] = 19;
    dimensions<3> dims(index_range<3>(i3a, i3b));

    sequence<3, unsigned> syma_grp, syma_sym;
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 0;
    syma_sym[0] = 0;
    ctf_symmetry<3, double> syma(syma_grp, syma_sym);

    dense_tensor<3, double, allocator_t> ta(dims), ta_ref(dims);
    ctf_dense_tensor<3, double> dta(dims, syma);

    ctf_tod_random<3>().perform(true, dta);
    ctf_tod_collect<3>(dta).perform(ta);

    tod_copy<3>(ta, permutation<3>().permute(0, 1)).perform(true, ta_ref);
    compare_ref<3>::compare(testname, ta, ta_ref, 1e-15);

    tod_copy<3>(ta, permutation<3>().permute(1, 2)).perform(true, ta_ref);
    compare_ref<3>::compare(testname, ta, ta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

