#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_mult.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_mult.h>
#include "../compare_ref.h"
#include "ctf_tod_mult_test.h"

namespace libtensor {


void ctf_tod_mult_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_mult_1a();
        test_mult_1b();
        test_div_1a();
        test_div_1b();
        test_2(true, false);
        test_2(false, false);
        test_2(true, true);
        test_2(false, true);

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_mult_test::test_mult_1a() {

    static const char testname[] = "ctf_tod_mult_test::test_mult_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);
    tod_mult<2>(ta, tb, false, 0.5).perform(true, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);
    ctf_tod_mult<2>(dta, dtb, false, 0.5).perform(true, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult_test::test_mult_1b() {

    static const char testname[] = "ctf_tod_mult_test::test_mult_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);
    tod_mult<2>(ta, tb, false, 0.5).perform(false, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);
    ctf_tod_mult<2>(dta, dtb, false, 0.5).perform(false, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult_test::test_div_1a() {

    static const char testname[] = "ctf_tod_mult_test::test_div_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);
    tod_mult<2>(ta, tb, true, 0.5).perform(true, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);
    ctf_tod_mult<2>(dta, dtb, true, 0.5).perform(true, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult_test::test_div_1b() {

    static const char testname[] = "ctf_tod_mult_test::test_div_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 49;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<2, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc);
    ctf_dense_tensor<2, double> dta(dimsa), dtb(dimsb), dtc(dimsc);

    tod_random<2>().perform(ta);
    tod_random<2>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);
    tod_mult<2>(ta, tb, true, 0.5).perform(false, tc_ref);

    ctf_tod_distribute<2>(ta).perform(dta);
    ctf_tod_distribute<2>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);
    ctf_tod_mult<2>(dta, dtb, true, 0.5).perform(false, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_mult_test::test_2(bool zero, bool muldiv) {

    std::ostringstream tnss;
    tnss << "ctf_tod_mult_test::test_2(" << zero << ", " << muldiv << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 15; i2[3] = 15;
    dimensions<4> dimsa(index_range<4>(i1, i2)), dimsb(dimsa), dimsc(dimsa);
    dense_tensor<4, double, allocator_t> ta(dimsa), tb(dimsb), tc(dimsc),
        tc_ref(dimsc), tt(dimsa);

    sequence<4, unsigned> syma_grp, syma_sym;
    syma_grp[0] = 0; syma_grp[1] = 0; syma_grp[2] = 1; syma_grp[3] = 1;
    syma_sym[0] = 0; syma_sym[1] = 0;
    ctf_symmetry<4, double> syma(syma_grp, syma_sym);
    ctf_symmetry<4, double> symb(syma_grp, syma_sym);
    syma_sym[0] = 1; syma_sym[1] = 1;
    syma.add_component(syma_grp, syma_sym);
    ctf_dense_tensor<4, double> dta(dimsa, syma), dtb(dimsb, symb),
        dtc(dimsc, syma);

    permutation<4> p0123, p0132, p1023, p1032;
    p0132.permute(2, 3);
    p1023.permute(0, 1);
    p1032.permute(0, 1).permute(2, 3);

    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, ta);
    tod_copy<4>(tt, p1032).perform(false, ta);
    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, tb);
    tod_copy<4>(tt, p1023).perform(false, tb);
    tod_copy<4>(tt, p0132).perform(false, tb);
    tod_copy<4>(tt, p1032).perform(false, tb);
    tod_random<4>().perform(tt);
    tod_copy<4>(tt).perform(true, tc);
    tod_copy<4>(tt, p1032).perform(false, tc);
    tod_copy<4>(tc).perform(true, tc_ref);
    tod_mult<4>(ta, tb, muldiv, 0.45).perform(zero, tc_ref);

    ctf_tod_distribute<4>(ta).perform(dta);
    ctf_tod_distribute<4>(tb).perform(dtb);
    ctf_tod_distribute<4>(tc).perform(dtc);
    ctf_tod_mult<4>(dta, dtb, muldiv, 0.45).perform(zero, dtc);
    ctf_tod_collect<4>(dtc).perform(tc);

    compare_ref<4>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

