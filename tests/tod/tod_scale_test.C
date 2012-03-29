#include <cmath>
#include <ctime>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/tod/tod_scale.h>
#include "../compare_ref.h"
#include "tod_scale_test.h"

namespace libtensor {


void tod_scale_test::perform() throw(libtest::test_exception) {

    srand48(time(0));

    test_0();
    test_i(1);
    test_i(3);
    test_i(16);
    test_ij(1, 1);
    test_ij(1, 3);
    test_ij(3, 1);
    test_ij(3, 7);
    test_ij(16, 16);
    test_ijkl(1, 1, 1, 1);
    test_ijkl(1, 1, 1, 3);
    test_ijkl(1, 3, 1, 3);
    test_ijkl(3, 5, 7, 11);
    test_ijkl(16, 16, 16, 16);
}


template<size_t N>
void tod_scale_test::test_generic(const char *testname,
    const dimensions<N> &d, double c) throw(libtest::test_exception) {

    typedef std_allocator<double> allocator_t;

    try {

    dense_tensor<N, double, allocator_t> t(d), t_ref(d);

    {
    dense_tensor_ctrl<N, double> tc(t), tc_ref(t_ref);
    double *p = tc.req_dataptr(), *p_ref = tc_ref.req_dataptr();
    size_t sz = d.get_size();

    for(size_t i = 0; i < sz; i++) {
        double a = drand48();
        p[i] = a;
        p_ref[i] = c * a;
    }

    tc.ret_dataptr(p); p = 0;
    tc_ref.ret_dataptr(p_ref); p_ref = 0;
    }

    tod_scale<N>(t, c).perform();

    compare_ref<N>::compare(testname, t, t_ref, 1e-15);
    
    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tod_scale_test::test_0() throw(libtest::test_exception) {

    static const char *testname = "tod_scale_test::test_0()";

    try {

    index<0> i1, i2;
    dimensions<0> dims(index_range<0>(i1, i2));
    test_generic(testname, dims, 1.0);
    test_generic(testname, dims, 0.0);
    test_generic(testname, dims, -0.5);
    test_generic(testname, dims, 2.3);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tod_scale_test::test_i(size_t i) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tod_scale_test::test_i(" << i << ")";
    std::string tn = ss.str();

    try {

    index<1> i1, i2;
    i2[0] = i - 1;
    dimensions<1> dims(index_range<1>(i1, i2));
    test_generic(tn.c_str(), dims, 1.0);
    test_generic(tn.c_str(), dims, 0.0);
    test_generic(tn.c_str(), dims, -0.5);
    test_generic(tn.c_str(), dims, 2.3);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_scale_test::test_ij(size_t i, size_t j)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tod_scale_test::test_ij(" << i << ", " << j << ")";
    std::string tn = ss.str();

    try {

    index<2> i1, i2;
    i2[0] = i - 1; i2[1] = j - 1;
    dimensions<2> dims(index_range<2>(i1, i2));
    test_generic(tn.c_str(), dims, 1.0);
    test_generic(tn.c_str(), dims, 0.0);
    test_generic(tn.c_str(), dims, -0.5);
    test_generic(tn.c_str(), dims, 2.3);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_scale_test::test_ijkl(size_t i, size_t j, size_t k, size_t l)
    throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "tod_scale_test::test_ijkl(" << i << ", " << j << ", "
        << k << ", " << l << ")";
    std::string tn = ss.str();

    try {

    index<4> i1, i2;
    i2[0] = i - 1; i2[1] = j - 1; i2[2] = k - 1; i2[3] = l - 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    test_generic(tn.c_str(), dims, 1.0);
    test_generic(tn.c_str(), dims, 0.0);
    test_generic(tn.c_str(), dims, -0.5);
    test_generic(tn.c_str(), dims, 2.3);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
