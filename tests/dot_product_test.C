#include <sstream>
#include <libtensor.h>
#include "dot_product_test.h"

namespace libtensor {

void dot_product_test::perform() throw(libtest::test_exception) {

	//test_1(10);
	test_2_ij_ij(10, 11);
	test_2_ij_ji(10, 11);

}

void dot_product_test::test_1(size_t ni) throw(libtest::test_exception) {
/*
	std::ostringstream testname;
	testname << "dot_product_test::test_1(" << ni << ")";

	bispace<1> sp_i(ni);
	btensor<1> t1(sp_i), t2(sp_i);
	letter i;

	double c;
	try {
		c = dot_product(t1(i), t2(i));
	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}
*/
}

void dot_product_test::test_2_ij_ij(size_t ni, size_t nj)
	throw(libtest::test_exception) {

	std::ostringstream testname;
	testname << "dot_product_test::test_2_ij_ij(" <<
		ni << ", " << nj << ")";

	bispace<1> sp_i(ni), sp_j(nj);
	bispace<2> sp_ij(sp_i*sp_j);
	btensor<2> bt1(sp_ij), bt2(sp_ij);
	letter i, j;

	index<2> i0;
	block_tensor_ctrl<2, double> btc1(bt1);
	block_tensor_ctrl<2, double> btc2(bt2);
	tensor_i<2, double> &t1 = btc1.req_block(i0);
	tensor_i<2, double> &t2 = btc2.req_block(i0);

	tensor_ctrl<2, double> tc1(t1);
	tensor_ctrl<2, double> tc2(t2);
	double *p1 = tc1.req_dataptr();
	double *p2 = tc2.req_dataptr();

	index<2> i1;
	dimensions<2> dims1(t1.get_dims());
	dimensions<2> dims2(t2.get_dims());
	do {
		size_t ii1 = dims1.abs_index(i1);
		size_t ii2 = dims2.abs_index(i1);
		p1[ii1] = drand48();
		p2[ii2] = drand48();
	} while(dims1.inc_index(i1));

	tc1.ret_dataptr(p1);
	tc2.ret_dataptr(p2);

	tod_dotprod<2> op_ref(t1, t2);
	double c_ref = op_ref.calculate();

	double c;
	try {
		c = dot_product(bt1(i|j), bt2(i|j));
	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}

	if(fabs(c - c_ref) > fabs(c_ref * k_thresh)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << c << " (result), "
			<< c_ref << " (reference), " << c - c_ref
			<< " (diff)";
		fail_test(testname.str().c_str(), __FILE__, __LINE__,
			ss.str().c_str());
	}

}

void dot_product_test::test_2_ij_ji(size_t ni, size_t nj)
	throw(libtest::test_exception) {

	std::ostringstream testname;
	testname << "dot_product_test::test_2_ij_ji(" <<
		ni << ", " << nj << ")";

	bispace<1> sp_i(ni), sp_j(nj);
	bispace<2> sp_ij(sp_i*sp_j), sp_ji(sp_j*sp_i);
	btensor<2> bt1(sp_ij), bt2(sp_ji);
	letter i, j;

	index<2> i0;
	block_tensor_ctrl<2, double> btc1(bt1);
	block_tensor_ctrl<2, double> btc2(bt2);
	tensor_i<2, double> &t1 = btc1.req_block(i0);
	tensor_i<2, double> &t2 = btc2.req_block(i0);

	tensor_ctrl<2, double> tc1(t1);
	tensor_ctrl<2, double> tc2(t2);
	double *p1 = tc1.req_dataptr();
	double *p2 = tc2.req_dataptr();

	index<2> i1;
	permutation<2> perm; perm.permute(0, 1);
	dimensions<2> dims1(t1.get_dims());
	dimensions<2> dims2(t2.get_dims());
	do {
		index<2> i2(i1); i2.permute(perm);
		size_t ii1 = dims1.abs_index(i1);
		size_t ii2 = dims2.abs_index(i2);
		p1[ii1] = drand48();
		p2[ii2] = drand48();
	} while(dims1.inc_index(i1));

	tc1.ret_dataptr(p1);
	tc2.ret_dataptr(p2);

	permutation<2> perm1, perm2; perm2.permute(0, 1);
	tod_dotprod<2> op_ref(t1, perm1, t2, perm2);
	double c_ref = op_ref.calculate();

	double c;
	try {
		c = dot_product(bt1(i|j), bt2(j|i));
	} catch(exception &e) {
		fail_test(testname.str().c_str(), __FILE__, __LINE__, e.what());
	}

	if(fabs(c - c_ref) > fabs(c_ref * k_thresh)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << c << " (result), "
			<< c_ref << " (reference), " << c - c_ref
			<< " (diff)";
		fail_test(testname.str().c_str(), __FILE__, __LINE__,
			ss.str().c_str());
	}

}

} // namespace libtensor
