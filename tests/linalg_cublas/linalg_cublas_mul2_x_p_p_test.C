#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/linalg/cublas/linalg_cublas.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_cublas_mul2_x_p_p_test.h"

namespace libtensor {


void linalg_cublas_mul2_x_p_p_test::perform() throw(libtest::test_exception) {

    test_mul2_x_p_p(1, 1, 1);
    test_mul2_x_p_p(2, 1, 1);
    test_mul2_x_p_p(16, 1, 1);
    test_mul2_x_p_p(17, 1, 1);
    test_mul2_x_p_p(2, 2, 3);
    test_mul2_x_p_p(2, 3, 2);
}


void linalg_cublas_mul2_x_p_p_test::test_mul2_x_p_p(size_t np, size_t spa,
    size_t spb) {

    std::ostringstream ss;
    ss << "linalg_cublas_mul2_x_p_p_test::test_mul2_x_p_p("
        << np << ", " << spa << ", " << spb << ")";
    std::string tnss = ss.str();

    typedef libvmm::cuda_allocator<double> cuda_allocator_type;
    typedef typename cuda_allocator_type::pointer_type cuda_pointer;

    try {

    size_t sza = np * spa, szb = np * spb;

    std::vector<double> va(sza, 0.0), vb(szb, 0.0);

    for(size_t i = 0; i < sza; i++) va[i] = drand48();
    for(size_t i = 0; i < szb; i++) vb[i] = drand48();

    const double *a = &va[0];
    const double *b = &vb[0];

    cuda_pointer pa1 = cuda_allocator_type::allocate(sza);
    cuda_pointer pb1 = cuda_allocator_type::allocate(szb);
    cuda_allocator_type::copy_to_device(pa1, a, sza);
    cuda_allocator_type::copy_to_device(pb1, b, szb);
    const double *a1 = cuda_allocator_type::lock_ro(pa1);
    const double *b1 = cuda_allocator_type::lock_ro(pb1);

    cublasHandle_t cbh;
    cublasStatus_t ec = cublasCreate(&cbh);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Failed cublasCreate().");
    }

    double c = linalg_cublas::mul2_x_p_p(cbh, np, a1, spa, b1, spb);
    double c_ref = linalg_generic::mul2_x_p_p(0, np, a, spa, b, spb);

    cublasDestroy(cbh);

    cuda_allocator_type::unlock_ro(pa1);
    cuda_allocator_type::unlock_ro(pb1);
    cuda_allocator_type::deallocate(pa1);
    cuda_allocator_type::deallocate(pb1);

    if(!cmp(c - c_ref, c_ref)) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
