#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/linalg/cublas/linalg_cublas.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_cublas_mul2_i_i_x_test.h"

namespace libtensor {


void linalg_cublas_mul2_i_i_x_test::perform() throw(libtest::test_exception) {

    test_mul2_i_i_x(1, 1, 1);
    test_mul2_i_i_x(2, 1, 1);
    test_mul2_i_i_x(16, 1, 1);
    test_mul2_i_i_x(17, 1, 1);
    test_mul2_i_i_x(2, 2, 3);
    test_mul2_i_i_x(2, 3, 2);
}


void linalg_cublas_mul2_i_i_x_test::test_mul2_i_i_x(size_t ni, size_t sia,
    size_t sic) {

    std::ostringstream ss;
    ss << "linalg_cublas_mul2_i_i_x_test::test_mul2_i_i_x("
        << ni << ", " << sia << ", " << sic << ")";
    std::string tnss = ss.str();

    typedef libvmm::cuda_allocator<double> cuda_allocator_type;
    typedef typename cuda_allocator_type::pointer_type cuda_pointer;

    try {

    size_t sza = ni * sia, szc = ni * sic;

    std::vector<double> va(sza, 0.0), vc(szc, 0.0), vc_ref(szc, 0.0);
    double b;

    for(size_t i = 0; i < sza; i++) va[i] = drand48();
    for(size_t i = 0; i < szc; i++) vc[i] = vc_ref[i] = drand48();
    b = drand48();

    const double *a = &va[0];
    double *c = &vc[0];
    double *c_ref = &vc_ref[0];

    cuda_pointer pa1 = cuda_allocator_type::allocate(sza);
    cuda_pointer pc1 = cuda_allocator_type::allocate(szc);
    cuda_allocator_type::copy_to_device(pa1, a, sza);
    cuda_allocator_type::copy_to_device(pc1, c, szc);
    const double *a1 = cuda_allocator_type::lock_ro(pa1);
    double *c1 = cuda_allocator_type::lock_rw(pc1);

    cublasHandle_t cbh;
    cublasStatus_t ec = cublasCreate(&cbh);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Failed cublasCreate().");
    }

    linalg_cublas::mul2_i_i_x(cbh, ni, a1, sia, b, c1, sic);
    linalg_generic::mul2_i_i_x(0, ni, a, sia, b, c_ref, sic);

    cuda_allocator_type::copy_to_host(c, pc1, szc);

    cublasDestroy(cbh);

    cuda_allocator_type::unlock_ro(pa1);
    cuda_allocator_type::unlock_rw(pc1);
    cuda_allocator_type::deallocate(pa1);
    cuda_allocator_type::deallocate(pc1);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(c[i] - c_ref[i], c_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
        }
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
