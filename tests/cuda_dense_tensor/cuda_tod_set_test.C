#include <libtensor/cuda_dense_tensor/cuda_dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_set.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_d2h.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>
#include "cuda_tod_set_test.h"
#include "../compare_ref.h"

namespace libtensor {


void cuda_tod_set_test::perform() throw(libtest::test_exception) {

    test_1(1.0);
    test_1(-0.5);
    test_1(0.0);
    test_2(1.0);
    test_2(-0.5);
    test_2(0.0);
}


void cuda_tod_set_test::test_1(double v) {

    std::ostringstream tnss;
    tnss << "cuda_tod_set_test::test_1(" << v << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 3; i2[1] = 3; i2[2] = 4; i2[3] = 4;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    cuda_dense_tensor<4, double, cuda_allocator_t> d_t(dim);
    dense_tensor<4, double, allocator_t> h_t(dim), h_t_ref(dim);

    tod_random<4>().perform(h_t);
    cuda_tod_copy_h2d<4>(h_t).perform(d_t);
    cuda_tod_set<4>(v).perform(d_t);
    cuda_tod_copy_d2h<4>(d_t).perform(h_t);

    tod_set<4>(v).perform(h_t_ref);

    compare_ref<4>::compare(tn.c_str(), h_t, h_t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_set_test::test_2(double v) {

    std::ostringstream tnss;
    tnss << "cuda_tod_set_test::test_2(" << v << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 17; i2[1] = 17; i2[2] = 17; i2[3] = 17;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    cuda_dense_tensor<4, double, cuda_allocator_t> d_t(dim);
    dense_tensor<4, double, allocator_t> h_t(dim), h_t_ref(dim);

    tod_random<4>().perform(h_t);
    cuda_tod_copy_h2d<4>(h_t).perform(d_t);
    cuda_tod_set<4>(v).perform(d_t);
    cuda_tod_copy_d2h<4>(d_t).perform(h_t);

    tod_set<4>(v).perform(h_t_ref);

    compare_ref<4>::compare(tn.c_str(), h_t, h_t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

