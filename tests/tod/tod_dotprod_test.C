#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_dotprod.h>
#include "tod_dotprod_test.h"

namespace libtensor {

const double tod_dotprod_test::k_thresh = 1e-14;

void tod_dotprod_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	test_1(4);
	test_1(16);
	test_1(200);

	permutation<2> p2;
	test_2(4, 4, p2);
	test_2(10, 20, p2);
	p2.permute(0, 1);
	test_2(10, 21, p2);

}

void tod_dotprod_test::test_1(size_t ni) throw(libtest::test_exception) {

	typedef std_allocator<double> allocator;

	std::ostringstream testname;
	testname << "tod_dotprod_test::test_1(" << ni << ")";

	try {

	index<1> ia1, ia2; ia2[0] = ni-1;
	index<1> ib1, ib2; ib2[0] = ni-1;
	dimensions<1> dima(index_range<1>(ia1, ia2));
	dimensions<1> dimb(index_range<1>(ib1, ib2));
	size_t sza = dima.get_size(), szb = dimb.get_size();

	tensor<1, double, allocator> ta(dima);
	tensor<1, double, allocator> tb(dimb);

	double c_ref = 0.0;
	{
	tensor_ctrl<1, double> tca(ta);
	tensor_ctrl<1, double> tcb(tb);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();

	// Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

	// Generate reference data

	for(size_t i = 0; i < sza; i++) c_ref += dta[i] * dtb[i];
	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	}

	// Invoke the operation

	tod_dotprod<1> op(ta, tb);
	double c = op.calculate();

	// Compare against the reference

	if(fabs(c - c_ref) > fabs(c_ref * k_thresh)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << c << " (result), "
			<< c_ref << " (reference), " << c - c_ref << " (diff)";
		fail_test(testname.str().c_str(), __FILE__, __LINE__,
			ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}

}

void tod_dotprod_test::test_2(size_t ni, size_t nj, const permutation<2> &perm)
	throw(libtest::test_exception) {

	typedef std_allocator<double> allocator;

	std::ostringstream testname;
	testname << "tod_dotprod_test::test_2(" << ni << ", " << nj << ")";

	try {

	index<2> ia1, ia2; ia2[0] = ni-1; ia2[1] = nj-1;
	index<2> ib1, ib2; ib2[0] = ni-1; ib2[1] = nj-1;
	dimensions<2> dima(index_range<2>(ia1, ia2));
	dimensions<2> dimb(index_range<2>(ib1, ib2));
	dimb.permute(perm);
	size_t sza = dima.get_size(), szb = dimb.get_size();

	tensor<2, double, allocator> ta(dima);
	tensor<2, double, allocator> tb(dimb);

	double c_ref = 0.0;
	{
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<2, double> tcb(tb);
	double *dta = tca.req_dataptr();
	double *dtb = tcb.req_dataptr();

	// Fill in random input

	for(size_t i = 0; i < sza; i++) dta[i] = drand48();
	for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

	// Generate reference data

	abs_index<2> aia(dima);
	do {
		index<2> ib(aia.get_index()); ib.permute(perm);
		abs_index<2> aib(ib, dimb);
		size_t i = aia.get_abs_index();
		size_t j = aib.get_abs_index();
		c_ref += dta[i] * dtb[j];
	} while(aia.inc());
	tca.ret_dataptr(dta); dta = NULL; ta.set_immutable();
	tcb.ret_dataptr(dtb); dtb = NULL; tb.set_immutable();
	}

	// Invoke the operation

	permutation<2> p0;
	tod_dotprod<2> op(ta, p0, tb, perm);
	double c = op.calculate();

	// Compare against the reference

	if(fabs(c - c_ref) > fabs(c_ref * k_thresh)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << c << " (result), "
			<< c_ref << " (reference), " << c - c_ref << " (diff)";
		fail_test(testname.str().c_str(), __FILE__, __LINE__,
			ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}

}

} // namespace libtensor
