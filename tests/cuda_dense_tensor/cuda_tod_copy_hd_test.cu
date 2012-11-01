#include <sstream>
#include <libvmm/cuda_allocator.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_d2h.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>
#include "../compare_ref.h"
#include "cuda_tod_copy_hd_test.h"
#include <iostream>

namespace libtensor {

typedef std_allocator<double> std_allocator_t;
typedef libvmm::cuda_allocator<double> cuda_allocator_t;
typedef dense_tensor<4, double, cuda_allocator_t> d_tensor4;

void cuda_tod_copy_hd_test::perform() throw(libtest::test_exception) {
	test_exc();

	index<2> i2a, i2b; i2b[0]=10; i2b[1]=12;
	index_range<2> ir2(i2a, i2b); dimensions<2> dims2(ir2);

	test_plain(dims2);

	index<6> i6a, i6b;
	i6b[0]=5; i6b[1]=4; i6b[2]=5; i6b[3]=4; i6b[4]=3; i6b[5]=5;
	index_range<6> ir6(i6a, i6b); dimensions<6> dims6(ir6);
	test_plain(dims6);

}

template<size_t N>
void cuda_tod_copy_hd_test::test_plain(const dimensions<N> &dims)
	throw(libtest::test_exception) {

	static const char *testname = "cuda_tod_copy_hd_test::test_plain()";

	try {

	dense_tensor<N, double, std_allocator_t> h_ta(dims), h_ta_copy(dims);
	dense_tensor<N, double, cuda_allocator_t> d_ta(dims);

	{
	dense_tensor_ctrl<N, double> h_tca(h_ta), h_tca_copy(h_ta_copy);
	dense_tensor_ctrl<N, double> d_tca(d_ta);

	double *h_dta = h_tca.req_dataptr();
	//double *h_dtb1 = h_tcb.req_dataptr();
	//double *d_dta = d_tca.req_dataptr();

	// Fill in random data
	abs_index<N> aida(dims);
	do {
		size_t i = aida.get_abs_index();
		h_dta[i] = drand48();
	} while(aida.inc());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	//copy a from host to device
	cuda_tod_copy_h2d<N>(h_ta).perform(d_ta);

	//copy a back to tensor h_ta_copy
	cuda_tod_copy_d2h<N>(d_ta).perform(h_ta_copy);

	}
	// Compare against the reference


	compare_ref<N>::compare(testname, h_ta, h_ta_copy, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}


void cuda_tod_copy_hd_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2, i3;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
	index_range<4> ir1(i1,i2), ir2(i1,i3);
	dimensions<4> dim1(ir1), dim2(ir2);
	d_tensor4 t1(dim1), t2(dim2);

	bool ok = false;
	try {
		cuda_tod_copy_h2d<4>(t1).perform(t2);
	} catch(exception &e) {
		ok = true;
	}

	try {
		cuda_tod_copy_d2h<4>(t1).perform(t2);
	} catch(exception &e) {
		ok = ok && true;
	}

	if(!ok) {
		fail_test("cuda_tod_copy_hd_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}
}

} // namespace libtensor

