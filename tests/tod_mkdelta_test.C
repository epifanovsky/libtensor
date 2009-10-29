#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor.h>
#include "compare_ref.h"
#include "tod_mkdelta_test.h"

namespace libtensor {


void tod_mkdelta_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_1(5, 14);
	test_1(3, 5);
}


void tod_mkdelta_test::test_1(size_t ni, size_t na)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_mkdelta_test::test_1(" << ni << ", " << na << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	// Create tensors: f_{ii}, f_{aa}, d_{ia}

	index<2> ii1, ii2, ia1, ia2, id1, id2;
	ii2[0] = ni - 1; ii2[1] = ni - 1;
	ia2[0] = na - 1; ia2[1] = na - 1;
	id2[0] = ni - 1; id2[1] = na - 1;
	dimensions<2> dimi(index_range<2>(ii1, ii2));
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimd(index_range<2>(id1, id2));
	tensor<2, double, allocator_t> ti(dimi), ta(dima), td(dimd),
		td_ref(dimd);
	tensor_ctrl<2, double> tci(ti), tca(ta), tcd(td), tcd_ref(td_ref);

	size_t szi = dimi.get_size();
	size_t sza = dima.get_size();
	size_t szd = dimd.get_size();
	double *dti = tci.req_dataptr();
	double *dta = tca.req_dataptr();
	double *dtd = tcd.req_dataptr();
	double *dtd_ref = tcd_ref.req_dataptr();

	// Fill in random input

	for(size_t i = 0; i < szi; i++) dti[i] = drand48();
	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szd; i++) dtd[i] = drand48();

	// Generate reference data

	index<2> ii, aa, ia;
	for(size_t i = 0; i < ni; i++) {
	for(size_t a = 0; a < na; a++) {
		ii[0] = i; ii[1] = i;
		aa[0] = a; aa[1] = a;
		ia[0] = i; ia[1] = a;
		abs_index<2> aii(ii, dimi), aaa(aa, dima), aia(ia, dimd);
		dtd_ref[aia.get_abs_index()] =
			dti[aii.get_abs_index()] - dta[aaa.get_abs_index()];
	}
	}

	tci.ret_dataptr(dti); dti = NULL;
	tca.ret_dataptr(dta); dta = NULL;
	tcd.ret_dataptr(dtd); dtd = NULL;
	tcd_ref.ret_dataptr(dtd_ref); dtd_ref = NULL;

	// Invoke the operation

	tod_mkdelta(ti, ta).perform(td);

	// Compare data

	compare_ref<2>::compare(tnss.str().c_str(), td, td_ref, 1e-15);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor

