#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_dirsum.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_dirsum_test.h"

namespace libtensor {


void ctf_tod_dirsum_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_dirsum_test::test_1a() {

    static const char testname[] = "ctf_tod_dirsum_test::test_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<2> dimsc(index_range<2>(i2a, i2b));

    dense_tensor<1, double, allocator_t> ta(dimsa);
    dense_tensor<1, double, allocator_t> tb(dimsb);
    dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
    ctf_dense_tensor<1, double> dta(dimsa);
    ctf_dense_tensor<1, double> dtb(dimsb);
    ctf_dense_tensor<2, double> dtc(dimsc);

    tod_random<1>().perform(ta);
    tod_random<1>().perform(tb);
    tod_random<2>().perform(tc);
    scalar_transf<double> ka(-1.0), kb(0.5);
    tensor_transf<2, double> trc(permutation<2>(), scalar_transf<double>(1.0));
    tod_dirsum<1, 1>(ta, ka, tb, kb, trc).perform(true, tc_ref);

    ctf_tod_distribute<1>(ta).perform(dta);
    ctf_tod_distribute<1>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);

    ctf_tod_dirsum<1, 1>(dta, ka, dtb, kb, trc).perform(true, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_dirsum_test::test_1b() {

    static const char testname[] = "ctf_tod_dirsum_test::test_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<2> dimsc(index_range<2>(i2a, i2b));

    dense_tensor<1, double, allocator_t> ta(dimsa);
    dense_tensor<1, double, allocator_t> tb(dimsb);
    dense_tensor<2, double, allocator_t> tc(dimsc), tc_ref(dimsc);
    ctf_dense_tensor<1, double> dta(dimsa);
    ctf_dense_tensor<1, double> dtb(dimsb);
    ctf_dense_tensor<2, double> dtc(dimsc);

    tod_random<1>().perform(ta);
    tod_random<1>().perform(tb);
    tod_random<2>().perform(tc);
    tod_copy<2>(tc).perform(true, tc_ref);
    scalar_transf<double> ka(-1.0), kb(0.5);
    tensor_transf<2, double> trc(permutation<2>(), scalar_transf<double>(1.0));
    tod_dirsum<1, 1>(ta, ka, tb, kb, trc).perform(false, tc_ref);

    ctf_tod_distribute<1>(ta).perform(dta);
    ctf_tod_distribute<1>(tb).perform(dtb);
    ctf_tod_distribute<2>(tc).perform(dtc);

    ctf_tod_dirsum<1, 1>(dta, ka, dtb, kb, trc).perform(false, dtc);
    ctf_tod_collect<2>(dtc).perform(tc);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

