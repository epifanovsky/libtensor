#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_delta_denom2.h>
#include "tod_delta_denom2_test.h"
#include "compare_ref.h"


namespace libtensor {


void tod_delta_denom2_test::perform() throw(libtest::test_exception) {

	test_operation(5, 10);
	test_exceptions();
}


void tod_delta_denom2_test::test_operation(size_t ni, size_t na)
	throw(libtest::test_exception) {

	static const char *testname = "tod_delta_denom2_test::test_operation()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	// Create tensors: d_{ia}, t_{ijab}

	index<2> id1, id2;
	index<4> it1, it2;
	id2[0] = ni - 1; id2[1] = na - 1;
	it2[0] = ni - 1; it2[1] = ni - 1; it2[2] = na - 1; it2[3] = na - 1;
	dimensions<2> dimd(index_range<2>(id1, id2));
	dimensions<4> dimt(index_range<4>(it1, it2));
	tensor<2, double, allocator_t> td(dimd);
	tensor<4, double, allocator_t> tt(dimt), tt_ref(dimt);

	// Allocate memory for input and reference output

	size_t szd = dimd.get_size();
	size_t szt = dimt.get_size();
	double *dtd = new double[szd];
	double *dtt = new double[szt];
	double *dtt_ref = new double[szt];

	// Fill in random input

	for(size_t i = 0; i < szd; i++) dtd[i] = drand48();
	for(size_t i = 0; i < szt; i++) dtt[i] = drand48();

	// Generate reference data

	index<2> iia, ijb;
	index<4> it, ir;
	for(size_t i = 0; i < ni; i++) {
	for(size_t j = 0; j < ni; j++) {
	for(size_t a = 0; a < na; a++) {
	for(size_t b = 0; b < na; b++) {

		iia[0] = i; iia[1] = a;
		ijb[0] = j; ijb[1] = b;
		it[0] = i; it[1] = j; it[2] = a; it[3] = b;
		dtt_ref[dimt.abs_index(it)] =
			dtt[dimt.abs_index(it)] /
			(dtd[dimd.abs_index(iia)] + dtd[dimd.abs_index(ijb)]);
	}
	}
	}
	}

	// Copy input and reference output to tensors

	tensor_ctrl<2, double> tcd(td);
	tensor_ctrl<4, double> tct(tt), tct_ref(tt_ref);
	double *ptr = tcd.req_dataptr();
	for(size_t i = 0; i < szd; i++) ptr[i] = dtd[i];
	tcd.ret_dataptr(ptr);
	ptr = tct_ref.req_dataptr();
	for(size_t i = 0; i < szt; i++) ptr[i] = dtt_ref[i];
	tct_ref.ret_dataptr(ptr);
	td.set_immutable(); tt_ref.set_immutable();
	ptr = tct.req_dataptr();
	for(size_t i = 0; i < szt; i++) ptr[i] = dtt[i];
	tct.ret_dataptr(ptr);

	delete [] dtd; delete [] dtt; delete [] dtt_ref;

	// Invoke the operation

	tod_delta_denom2(td, td).perform(tt);

	// Compare data

	compare_ref<4>::compare(testname, tt, tt_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void tod_delta_denom2_test::test_exceptions() throw(libtest::test_exception) {
}

} // namespace libtensor
