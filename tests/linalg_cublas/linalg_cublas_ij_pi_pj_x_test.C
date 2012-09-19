#include <sstream>
#include <vector>
#include <cublas_v2.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/linalg/cublas/linalg_cublas.h>
#include <libtensor/linalg/generic/linalg_base_generic.h>
#include <libtensor/exception.h>
#include "linalg_cublas_ij_pi_pj_x_test.h"

namespace libtensor {


void linalg_cublas_ij_pi_pj_x_test::perform() throw(libtest::test_exception) {

	 test_ij_pi_pj_x(1, 1, 1, 1, 1, 1);
	test_ij_pi_pj_x(1, 2, 3, 2, 1, 2);
	test_ij_pi_pj_x(2, 1, 3, 1, 2, 1);
	test_ij_pi_pj_x(16, 16, 1, 16, 16, 16);
	test_ij_pi_pj_x(3, 17, 2, 17, 3, 17);
	test_ij_pi_pj_x(2, 2, 2, 2, 3, 4);
	test_ij_pi_pj_x(2, 2, 2, 4, 3, 2);
}


void linalg_cublas_ij_pi_pj_x_test::test_ij_pi_pj_x(size_t ni, size_t nj, size_t np,
	    size_t sic, size_t spa, size_t spb) {

	std::ostringstream ss;
	ss << "linalg_cublas_ij_pi_pj_x_test::test_ij_pi_pj_x("
		<< ni << ", " << nj << ", " << np << ", " << spa << ", "
		<< sic << ", " << spb << ")";
	std::string tnss = ss.str();

    typedef libvmm::cuda_allocator<double> cuda_allocator_type;
    typedef typename cuda_allocator_type::pointer_type cuda_pointer;


    try {

    size_t sza = np * spa, szb = np * spb, szc = ni * sic;

    std::vector<double> va(sza, 0.0), vb(szb, 0.0), vc(szc, 0.0), vc_ref(szc, 0.0);
	double d = drand48();

    for(size_t i = 0; i < sza; i++) va[i] = drand48();
	for(size_t i = 0; i < szb; i++) vb[i] = drand48();
	for(size_t i = 0; i < szc; i++) vc[i] = vc_ref[i] = drand48();

    const double *pa = &va[0];
    const double *pb = &vb[0];
    double *pc = &vc[0];
    double *pc_ref = &vc_ref[0];

    cuda_pointer pad = cuda_allocator_type::allocate(sza);
    cuda_pointer pbd = cuda_allocator_type::allocate(szb);
    cuda_pointer pcd = cuda_allocator_type::allocate(szc);
    cuda_allocator_type::copy_to_device(pad, pa, sza);
    cuda_allocator_type::copy_to_device(pbd, pb, szb);
    cuda_allocator_type::copy_to_device(pcd, pc, szc);
    const double *padl = cuda_allocator_type::lock_ro(pad);
    const double *pbdl = cuda_allocator_type::lock_ro(pbd);
    double *pcdl = cuda_allocator_type::lock_rw(pcd);

    cublasHandle_t cbh;
    cublasStatus_t ec = cublasCreate(&cbh);
    if(ec != cudaSuccess) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, "Failed cublasCreate().");
    }

    linalg_cublas::ij_pi_pj_x(cbh, ni, nj, np, padl, spa, pbdl, spb, pcdl, sic,
    		d);
    linalg_base_generic::ij_pi_pj_x(ni, nj, np, pa, spa, pb, spb, pc_ref, sic, d);

    cuda_allocator_type::copy_to_host(pc, pcdl, szc);

    cublasDestroy(cbh);

    cuda_allocator_type::unlock_ro(pad);
    cuda_allocator_type::unlock_ro(pbd);
    cuda_allocator_type::unlock_rw(pcd);
    cuda_allocator_type::deallocate(pad);
    cuda_allocator_type::deallocate(pbd);
    cuda_allocator_type::deallocate(pcd);

    for(size_t i = 0; i < szc; i++) {
        if(!cmp(vc[i] - vc_ref[i], vc_ref[i])) {
            fail_test(tnss.c_str(), __FILE__, __LINE__, "Incorrect result.");
        }
    }

    } catch(exception &e) {
        fail_test(tnss.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
