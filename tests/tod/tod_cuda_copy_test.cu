#include <sstream>
#include <libvmm/cuda_allocator.cu>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_cuda_copy.h>
#include "../compare_ref.h"
#include "tod_cuda_copy_test.h"
#include <iostream>

namespace libtensor {

typedef libvmm::std_allocator<double> std_allocator;
typedef libvmm::cuda_allocator<double> cuda_allocator;
typedef tensor<4, double, std_allocator> h_tensor4;
typedef tensor<4, double, cuda_allocator> d_tensor4;
typedef tensor_ctrl<4,double> tensor4_ctrl;

void tod_cuda_copy_test::perform() throw(libtest::test_exception) {
	test_exc();

	index<2> i2a, i2b; i2b[0]=10; i2b[1]=12;
	index_range<2> ir2(i2a, i2b); dimensions<2> dims2(ir2);
	permutation<2> perm2, perm2t;
	perm2t.permute(0, 1);

	test_plain(dims2);
	test_plain_additive(dims2, 1.0);
	test_plain_additive(dims2, -1.0);
	test_plain_additive(dims2, 2.5);

	test_scaled(dims2, 1.0);
	test_scaled(dims2, 0.5);
	test_scaled(dims2, -3.14);
	test_scaled_additive(dims2, 1.0, 1.0);
	test_scaled_additive(dims2, 0.5, 1.0);
	test_scaled_additive(dims2, -3.14, 1.0);
	test_scaled_additive(dims2, 1.0, -1.0);
	test_scaled_additive(dims2, 0.5, -1.0);
	test_scaled_additive(dims2, -3.14, -1.0);
	test_scaled_additive(dims2, 1.0, 2.5);
	test_scaled_additive(dims2, 0.5, 2.5);
	test_scaled_additive(dims2, -3.14, 2.5);

	test_perm(dims2, perm2);
	test_perm(dims2, perm2t);
	test_perm_additive(dims2, perm2, 1.0);
	test_perm_additive(dims2, perm2, -1.0);
	test_perm_additive(dims2, perm2, 2.5);
	test_perm_additive(dims2, perm2t, 1.0);
	test_perm_additive(dims2, perm2t, -1.0);
	test_perm_additive(dims2, perm2t, 2.5);

	test_perm_scaled(dims2, perm2, 1.0);
	test_perm_scaled(dims2, perm2t, 1.0);
	test_perm_scaled(dims2, perm2, 0.5);
	test_perm_scaled(dims2, perm2t, 0.5);
	test_perm_scaled(dims2, perm2, -3.14);
	test_perm_scaled(dims2, perm2t, -3.14);
	test_perm_scaled_additive(dims2, perm2, 1.0, 1.0);
	test_perm_scaled_additive(dims2, perm2t, 1.0, 1.0);
	test_perm_scaled_additive(dims2, perm2, 0.5, 1.0);
	test_perm_scaled_additive(dims2, perm2t, 0.5, 1.0);
	test_perm_scaled_additive(dims2, perm2, -3.14, 1.0);
	test_perm_scaled_additive(dims2, perm2t, -3.14, 1.0);
	test_perm_scaled_additive(dims2, perm2, 1.0, -1.0);
	test_perm_scaled_additive(dims2, perm2t, 1.0, -1.0);
	test_perm_scaled_additive(dims2, perm2, 0.5, -1.0);
	test_perm_scaled_additive(dims2, perm2t, 0.5, -1.0);
	test_perm_scaled_additive(dims2, perm2, -3.14, -1.0);
	test_perm_scaled_additive(dims2, perm2t, -3.14, -1.0);
	test_perm_scaled_additive(dims2, perm2, 1.0, 2.5);
	test_perm_scaled_additive(dims2, perm2t, 1.0, 2.5);
	test_perm_scaled_additive(dims2, perm2, 0.5, 2.5);
	test_perm_scaled_additive(dims2, perm2t, 0.5, 2.5);
	test_perm_scaled_additive(dims2, perm2, -3.14, 2.5);
	test_perm_scaled_additive(dims2, perm2t, -3.14, 2.5);

	index<6> i6a, i6b;
	i6b[0]=5; i6b[1]=4; i6b[2]=5; i6b[3]=4; i6b[4]=3; i6b[5]=5;
	index_range<6> ir6(i6a, i6b); dimensions<6> dims6(ir6);
	permutation<6> perm6, perm6t;
	perm6t.permute(0, 1).permute(1, 2);

//	index<4> i4a, i4b;
//	i4b[0] = 5; i4b[1] = 4; i4b[2] = 5; i4b[3] = 4;
////	i4b[0] = 1; i4b[1] = 2; i4b[2] = 3; i4b[3] = 4;
//	dimensions<4> dims2(index_range<4>(i4a, i4b));
//	permutation<4> perm2, perm2t;
//	perm2t.permute(0, 1).permute(1, 2).permute(2, 3);
//	perm2t.permute(0, 1).permute(1, 2);

	test_plain(dims6);
	test_plain_additive(dims6, 1.0);
	test_plain_additive(dims6, -1.0);
	test_plain_additive(dims6, 2.5);

	test_scaled(dims6, 1.0);
	test_scaled(dims6, 0.5);
	test_scaled(dims6, -3.14);
	test_scaled_additive(dims6, 1.0, 1.0);
	test_scaled_additive(dims6, 0.5, 1.0);
	test_scaled_additive(dims6, -3.14, 1.0);
	test_scaled_additive(dims6, 1.0, -1.0);
	test_scaled_additive(dims6, 0.5, -1.0);
	test_scaled_additive(dims6, -3.14, -1.0);
	test_scaled_additive(dims6, 1.0, 2.5);
	test_scaled_additive(dims6, 0.5, 2.5);
	test_scaled_additive(dims6, -3.14, 2.5);

	test_perm(dims6, perm6);
	test_perm(dims6, perm6t);
	test_perm_additive(dims6, perm6, 1.0);
	test_perm_additive(dims6, perm6, -1.0);
	test_perm_additive(dims6, perm6, 2.5);
	test_perm_additive(dims6, perm6t, 1.0);
	test_perm_additive(dims6, perm6t, -1.0);
	test_perm_additive(dims6, perm6t, 2.5);

	test_perm_scaled(dims6, perm6, 1.0);
	test_perm_scaled(dims6, perm6t, 1.0);
	test_perm_scaled(dims6, perm6, 0.5);
	test_perm_scaled(dims6, perm6t, 0.5);
	test_perm_scaled(dims6, perm6, -3.14);
	test_perm_scaled(dims6, perm6t, -3.14);
	test_perm_scaled_additive(dims6, perm6, 1.0, 1.0);
	test_perm_scaled_additive(dims6, perm6t, 1.0, 1.0);
	test_perm_scaled_additive(dims6, perm6, 0.5, 1.0);
	test_perm_scaled_additive(dims6, perm6t, 0.5, 1.0);
	test_perm_scaled_additive(dims6, perm6, -3.14, 1.0);
	test_perm_scaled_additive(dims6, perm6t, -3.14, 1.0);
	test_perm_scaled_additive(dims6, perm6, 1.0, -1.0);
	test_perm_scaled_additive(dims6, perm6t, 1.0, -1.0);
	test_perm_scaled_additive(dims6, perm6, 0.5, -1.0);
	test_perm_scaled_additive(dims6, perm6t, 0.5, -1.0);
	test_perm_scaled_additive(dims6, perm6, -3.14, -1.0);
	test_perm_scaled_additive(dims6, perm6t, -3.14, -1.0);
	test_perm_scaled_additive(dims6, perm6, 1.0, 2.5);
	test_perm_scaled_additive(dims6, perm6t, 1.0, 2.5);
	test_perm_scaled_additive(dims6, perm6, 0.5, 2.5);
	test_perm_scaled_additive(dims6, perm6t, 0.5, 2.5);
	test_perm_scaled_additive(dims6, perm6, -3.14, 2.5);
	test_perm_scaled_additive(dims6, perm6t, -3.14, 2.5);

	index<4> i4a, i4b;
	i4b[0] = 4; i4b[1] = 5; i4b[2] = 6; i4b[3] = 7;
	dimensions<4> dims4(index_range<4>(i4a, i4b));
	permutation<4> perm4, perm4c;
	perm4c.permute(0, 1).permute(1, 2).permute(2, 3);
////
	test_perm(dims4, perm4);
	test_perm(dims4, perm4c);

	test_perm_scaled(dims4, perm4, 1.0);
	test_perm_scaled(dims4, perm4c, 1.0);

	test_perm_scaled_additive(dims4, perm4c, 0.5, 2.5);
	test_perm_scaled_additive(dims4, perm4, -3.14, 2.5);
	test_perm_scaled_additive(dims4, perm4c, -3.14, 2.5);

}

template<size_t N>
void tod_cuda_copy_test::test_plain(const dimensions<N> &dims)
	throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_plain()";

	try {

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dims), h_tb_ref(dims);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dims);

	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data
	abs_index<N> aida(dims);
	do {
		size_t i = aida.get_abs_index();
		h_dta[i] = h_dtb2[i] = drand48();
		h_dtb1[i] = drand48();
	} while(aida.inc());

//	index<N> ida;
//	do {
//		size_t i;
//		i = dims.abs_index(ida);
//		h_dta[i] = h_dtb2[i] = drand48();
//		h_dtb1[i] = drand48();
//	} while(dims.inc_index(ida));

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	std::cout << "\nFirst adapter return\n ";
	}

	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta);
	std::cout << "\nCuda copy initialized\n ";
	cp.perform(d_tb);
	std::cout << "\nCuda copy performed\n ";

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb), h_tcb_ref(h_tb_ref);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();
		std::cout << "\nSeconf adapter requested\n ";
//		const double *h_dtb2 = h_tcb_ref.req_const_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

//		std::cout << "h_tb: " << h_dtb1[0] << ", " << h_dtb1[1] << ", " << h_dtb1[2] << ", " << h_dtb1[3]<< ", " << h_dtb1[4]<< ", " << h_dtb1[4]<< ", " << h_dtb1[6] << ", " << "\n";
//		std::cout << "h_tb_ref: " << h_dtb2[0] << ", " << h_dtb2[1] << ", " << h_dtb2[2] << ", " << h_dtb2[3]<< ", " << h_dtb2[4]<< ", " << h_dtb2[5]<< ", " << h_dtb2[6] << ", " << "\n";

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		std::cout << "Second adapter return\n ";
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
		std::cout << "Third adapter return\n ";
//		h_tcb_ref.ret_const_dataptr(h_dtb2); h_dtb2 = NULL;
	}


	// Compare against the reference


	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_plain_additive(const dimensions<N> &dims, double d)
	throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_plain_additive()";

	try {

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dims), h_tb_ref(dims);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dims);


	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data

	abs_index<N> aida(dims);
	do {
		size_t i;
		i = aida.get_abs_index();
		h_dta[i] = drand48();
		h_dtb1[i] = drand48();
		h_dtb2[i] = h_dtb1[i] + d * h_dta[i];
	} while(aida.inc());

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}

	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta);
	cp.perform(d_tb, d);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}


	// Compare against the reference

	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_scaled(const dimensions<N> &dims, double c)
	throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_scaled()";

	try {

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dims), h_tb_ref(dims);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dims);


	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data

	abs_index<N> aida(dims);
	do {
		size_t i;
		i = aida.get_abs_index();
		h_dta[i] = h_dtb2[i] = drand48();
		h_dtb2[i] *= c;
		h_dtb1[i] = drand48();
	} while(aida.inc());
	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, c);
	cp.perform(d_tb);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	std::ostringstream ss; ss << "tod_cuda_copy_test::test_scaled(" << c << ")";
	compare_ref<N>::compare(ss.str().c_str(), h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_scaled_additive(const dimensions<N> &dims, double c,
	double d) throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_scaled_additive()";

	try {

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dims), h_tb_ref(dims);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dims);


	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data

	abs_index<N> aida(dims);
	do {
		size_t i;
		i = aida.get_abs_index();
		h_dta[i] = drand48();
		h_dtb1[i] = drand48();
		h_dtb2[i] = h_dtb1[i] + c*d*h_dta[i];
	} while(aida.inc());

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, c);
	cp.perform(d_tb, d);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	std::ostringstream ss; ss << "tod_cuda_copy_test::test_scaled_additive("
		<< c << ")";
	compare_ref<N>::compare(ss.str().c_str(), h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_perm(const dimensions<N> &dims,
	const permutation<N> &perm) throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_perm()";

	try {

	dimensions<N> dimsa(dims), dimsb(dims);
	dimsb.permute(perm);

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dimsb), h_tb_ref(dimsb);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dimsb);


	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data
	abs_index<N> aida(dims);
	do {
		index<N> ida(aida.get_index());
		index<N> idb(ida);
		idb.permute(perm);
		abs_index<N> aidb(idb, dimsb);
		size_t i, j;
		i = aida.get_abs_index();
		j = aidb.get_abs_index();
		h_dta[i] = h_dtb2[j] = drand48();
		h_dtb1[i] = drand48();
	} while(aida.inc());

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, perm);
	cp.perform(d_tb);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_perm_additive(const dimensions<N> &dims,
	const permutation<N> &perm, double d) throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_perm_additive()";

	try {

		dimensions<N> dimsa(dims), dimsb(dims);
		dimsb.permute(perm);

		tensor<N, double, std_allocator> h_ta(dims), h_tb(dimsb), h_tb_ref(dimsb);
		tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dimsb);



	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data
	abs_index<N> aida(dims);
	do {
		index<N> ida(aida.get_index());
		index<N> idb(ida);
		idb.permute(perm);
		abs_index<N> aidb(idb, dimsb);
		size_t i, j;
		i = aida.get_abs_index();
		j = aidb.get_abs_index();
		h_dta[i] = drand48();
		h_dtb1[j] = drand48();
		h_dtb2[j] = h_dtb1[j] + d*h_dta[i];
	} while(aida.inc());

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, perm);
	cp.perform(d_tb, d);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_perm_scaled(const dimensions<N> &dims,
	const permutation<N> &perm, double c) throw(libtest::test_exception) {

	static const char *testname = "tod_cuda_copy_test::test_perm_scaled()";

	try {

	dimensions<N> dimsa(dims), dimsb(dims);
	dimsb.permute(perm);

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dimsb), h_tb_ref(dimsb);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dimsb);



	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data

	abs_index<N> aida(dims);
	do {
		index<N> ida(aida.get_index());
		index<N> idb(ida);
		idb.permute(perm);
		abs_index<N> aidb(idb, dimsb);
		size_t i, j;
		i = aida.get_abs_index();
		j = aidb.get_abs_index();
		h_dta[i] = drand48();
		h_dtb1[j] = drand48();
		h_dtb2[j] = c*h_dta[i];
	} while(aida.inc());

	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, perm, c);
	cp.perform(d_tb);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

	} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

template<size_t N>
void tod_cuda_copy_test::test_perm_scaled_additive(const dimensions<N> &dims,
	const permutation<N> &perm, double c, double d)
	throw(libtest::test_exception) {

	static const char *testname =
		"tod_cuda_copy_test::test_perm_scaled_additive()";

	try {

	dimensions<N> dimsa(dims), dimsb(dims);
	dimsb.permute(perm);

	tensor<N, double, std_allocator> h_ta(dims), h_tb(dimsb), h_tb_ref(dimsb);
	tensor<N, double, cuda_allocator> d_ta(dims), d_tb(dimsb);


	{
	tensor_ctrl<N, double> h_tca(h_ta), h_tcb(h_tb), h_tcb_ref(h_tb_ref);
	tensor_ctrl<N, double> d_tca(d_ta), d_tcb(d_tb);

	double *h_dta = h_tca.req_dataptr();
	double *h_dtb1 = h_tcb.req_dataptr();
	double *h_dtb2 = h_tcb_ref.req_dataptr();
	double *d_dta = d_tca.req_dataptr();
	double *d_dtb1 = d_tcb.req_dataptr();

	// Fill in random data

	abs_index<N> aida(dims);
	do {
		index<N> ida(aida.get_index());
		index<N> idb(ida);
		idb.permute(perm);
		abs_index<N> aidb(idb, dimsb);
		size_t i, j;
		i = aida.get_abs_index();
		j = aidb.get_abs_index();
		h_dta[i] = drand48();
		h_dtb1[j] = drand48();
		h_dtb2[j] = h_dtb1[j] + c*d*h_dta[i];
	} while(aida.inc());


	//copy a and b from host to device
	cuda_allocator::copy_to_device(d_dta, h_dta, dims.get_size());
	cuda_allocator::copy_to_device(d_dtb1, h_dtb1, dims.get_size());

	h_tca.ret_dataptr(h_dta); h_dta = NULL;
	h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
	h_tcb_ref.ret_dataptr(h_dtb2); h_dtb2 = NULL;
	d_tca.ret_dataptr(d_dta); d_dta = NULL;
	d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	h_ta.set_immutable(); h_tb_ref.set_immutable();
	}
	// Invoke the copy operation

	tod_cuda_copy<N> cp(d_ta, perm, c);
	cp.perform(d_tb, d);

	//copy from device to host
	{
		tensor_ctrl<N, double> h_tcb(h_tb);
		tensor_ctrl<N, double> d_tcb(d_tb);

		double *h_dtb1 = h_tcb.req_dataptr();
		double *d_dtb1 = d_tcb.req_dataptr();

		cuda_allocator::copy_to_host(h_dtb1, d_dtb1, dims.get_size());

		h_tcb.ret_dataptr(h_dtb1); h_dtb1 = NULL;
		d_tcb.ret_dataptr(d_dtb1); d_dtb1 = NULL;
	}

	// Compare against the reference

	compare_ref<N>::compare(testname, h_tb, h_tb_ref, 1e-15);

		} catch(exception &exc) {
		fail_test(testname, __FILE__, __LINE__, exc.what());
	}
}

void tod_cuda_copy_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2, i3;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
	index_range<4> ir1(i1,i2), ir2(i1,i3);
	dimensions<4> dim1(ir1), dim2(ir2);
	d_tensor4 t1(dim1), t2(dim2);

	bool ok = false;
	try {
		tod_cuda_copy<4> tc(t1); tc.perform(t2);
	} catch(exception &e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_cuda_copy_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}
}

} // namespace libtensor

