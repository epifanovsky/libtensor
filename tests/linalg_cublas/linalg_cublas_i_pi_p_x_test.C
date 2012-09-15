#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/linalg/cublas/linalg_cublas.h>
#include <libtensor/linalg/generic/linalg_base_generic.h>
#include <libtensor/exception.h>
#include "linalg_cublas_i_ip_p_x_test.h"

namespace libtensor {


void linalg_cublas_i_ip_p_x_test::perform() throw(libtest::test_exception) {

	 test_i_pi_p_x(1, 1, 1, 1, 1);
	    test_i_pi_p_x(1, 2, 1, 1, 1);
	    test_i_pi_p_x(2, 1, 1, 2, 1);
	    test_i_pi_p_x(16, 16, 1, 16, 1);
	    test_i_pi_p_x(17, 3, 1, 17, 1);
	    test_i_pi_p_x(2, 2, 2, 3, 4);
	    test_i_pi_p_x(2, 2, 4, 3, 2);
	}


void linalg_cublas_i_pi_p_x_test::test_i_pi_p_x(size_t ni, size_t np, size_t sic,
	size_t spa, size_t spb) {

	std::ostringstream ss;
	ss << "linalg_cublas_i_pi_p_x_test::test_i_pi_p_x("
		<< ni << ", " << np << ", " << sic << ", " << spa << ", "
		<< spb << ")";
	std::string tnss = ss.str();

    typedef libvmm::cuda_allocator<double> cuda_allocator_type;
    typedef typename cuda_allocator_type::pointer_type cuda_pointer;

    try {

	 size_t sza = np * spa, szb = np * spb, szc = ni * sic;

	a = new double[sza];
	b = new double[szb];
	c = new double[szc];
	c_ref = new double[szc];

    for(size_t i = 0; i < sza; i++) a[i] = drand48();
	for(size_t i = 0; i < szb; i++) b[i] = drand48();
	for(size_t i = 0; i < szc; i++) c[i] = c_ref[i] = drand48();

    const double *pa = &a[0];
    const double *pb = &b[0];
    double *pc = &c[0];
    double *pc_ref = &c_ref[0];

    cuda_pointer pad = cuda_allocator_type::allocate(sza);
    cuda_pointer pbd = cuda_allocator_type::allocate(szb);
    cuda_pointer pcd = cuda_allocator_type::allocate(szc);
    cuda_allocator_type::copy_to_device(pad, pa, sza);
    cuda_allocator_type::copy_to_device(pbd, pb, szb);
    cuda_allocator_type::copy_to_device(pcd, pc, szc);
    const double *padl = cuda_allocator_type::lock_ro(pad);
    const double *pbdl = cuda_allocator_type::lock_ro(pbd);
    double *pcdl = cuda_allocator_type::lock_rw(pcdl);

    cublasHandle_t cbh;
    cublasStatus_t ec = cublasCreate(&cbh);
    if(ec != cudaSuccess) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Failed cublasCreate().");
    }

    linalg_cublas::i_pi_p_x(cbh, ni, np, padl, spa, pbdl, spb, pcdl, sic);
    linalg_base_generic::i_pi_p_x(ni, np, a, spa, b, spb, c_ref, sic, d);

    cuda_allocator_type::copy_to_host(pc, pcdl, szc);

    cublasDestroy(cbh);

    cuda_allocator_type::unlock_ro(padl);
    cuda_allocator_type::unlock_ro(pbdl);
    cuda_allocator_type::unlock_rw(pcdl);
    cuda_allocator_type::deallocate(pad);
    cuda_allocator_type::deallocate(pbd);
    cuda_allocator_type::deallocate(pcd);

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
