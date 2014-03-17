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

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_mult_test::test_mult_1a() {

    static const char testname[] = "ctf_tod_mult_test::test_mult_1a()";

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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


} // namespace libtensor

