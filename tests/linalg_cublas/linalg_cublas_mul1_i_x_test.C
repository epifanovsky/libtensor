#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/linalg/cublas/linalg_cublas.h>
#include <libtensor/linalg/generic/linalg_generic.h>
#include <libtensor/exception.h>
#include "linalg_cublas_mul1_i_x_test.h"

namespace libtensor {


void linalg_cublas_mul1_i_x_test::perform() throw(libtest::test_exception) {

    test_mul1_i_x(1, 1);
    test_mul1_i_x(2, 1);
    test_mul1_i_x(1, 16);
    test_mul1_i_x(2, 3);
}


void linalg_cublas_mul1_i_x_test::test_mul1_i_x(size_t ni, size_t sic) {

    std::ostringstream ss;
    ss << "linalg_cublas_mul1_i_x_test::test_mul1_i_x("
        << ni  << ", " << sic << ")";
    std::string tnss = ss.str();

    typedef libvmm::cuda_allocator<double> cuda_allocator_type;
    typedef typename cuda_allocator_type::pointer_type cuda_pointer;

    try {

    size_t szc = ni * sic;

    std::vector<double> vc(szc, 0.0), vc_ref(szc, 0.0);
    double b;

    for(size_t i = 0; i < szc; i++) vc[i] = vc_ref[i] = drand48();
    b = drand48();

    double *c = &vc[0];
    double *c_ref = &vc_ref[0];

    cuda_pointer pc1 = cuda_allocator_type::allocate(szc);
    cuda_allocator_type::copy_to_device(pc1, c, szc);
    double *c1 = cuda_allocator_type::lock_rw(pc1);

    cublasHandle_t cbh;
    cublasStatus_t ec = cublasCreate(&cbh);
    if(ec != CUBLAS_STATUS_SUCCESS) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Failed cublasCreate().");
    }

    linalg_cublas::mul1_i_x(cbh, ni, b, c1, sic);
    linalg_generic::mul1_i_x(0, ni, b, c_ref, sic);

    cuda_allocator_type::copy_to_host(c, pc1, szc);

    cublasDestroy(cbh);

    cuda_allocator_type::unlock_rw(pc1);
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
