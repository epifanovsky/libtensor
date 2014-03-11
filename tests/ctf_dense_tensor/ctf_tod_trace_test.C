#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_trace.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_dense_tensor/ctf_dense_tensor.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_trace.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_distribute.h>
#include "../compare_ref.h"
#include "ctf_tod_trace_test.h"

namespace libtensor {


void ctf_tod_trace_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2();
        test_3();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_tod_trace_test::test_1() {

    static const char testname[] = "ctf_tod_trace_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> ta(dimsa);
    ctf_dense_tensor<2, double> dta(dimsa);

    tod_random<2>().perform(ta);

    ctf_tod_distribute<2>(ta).perform(dta);

    double d_ref = tod_trace<1>(ta).calculate();
    double d = ctf_tod_trace<1>(dta).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_trace_test::test_2() {

    static const char testname[] = "ctf_tod_trace_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 19; i2[3] = 9;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> ta(dimsa);
    ctf_dense_tensor<4, double> dta(dimsa);

    tod_random<4>().perform(ta);

    ctf_tod_distribute<4>(ta).perform(dta);

    permutation<4> perma;
    double d_ref = tod_trace<2>(ta, perma).calculate();
    double d = ctf_tod_trace<2>(dta, perma).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_tod_trace_test::test_3() {

    static const char testname[] = "ctf_tod_trace_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 9; i2[3] = 9;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    dense_tensor<4, double, allocator_t> ta(dimsa);
    ctf_dense_tensor<4, double> dta(dimsa);

    tod_random<4>().perform(ta);

    ctf_tod_distribute<4>(ta).perform(dta);

    permutation<4> perma;
    perma.permute(1, 2);
    double d_ref = tod_trace<2>(ta, perma).calculate();
    double d = ctf_tod_trace<2>(dta, perma).calculate();

    if(fabs(d - d_ref) > 1e-14 * fabs(d_ref)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (result), "
            << d_ref << " (reference), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

