#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/btod/btod_random.h>
#include <libtensor/iface/iface.h>
#include "dot_product_test.h"

namespace libtensor {


void dot_product_test::perform() throw(libtest::test_exception) {

	test_tt_ij_ij_1();
	test_tt_ij_ji_1();
	test_te_ij_ij_1();
	test_te_ij_ji_1();
}


void dot_product_test::test_tt_ij_ij_1() throw(libtest::test_exception) {

	static const char *testname = "dot_product_test::test_tt_ij_ij_1()";

	try {

	bispace<1> si(10), sj(11);
	bispace<2> sij(si|sj);
	btensor<2> bt1(sij), bt2(sij);

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	double c_ref = btod_dotprod<2>(bt1, bt2).calculate();

	letter i, j;
	double c = dot_product(bt1(i|j), bt2(i|j));
	check_ref(testname, c, c_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void dot_product_test::test_tt_ij_ji_1() throw(libtest::test_exception) {

	static const char *testname = "dot_product_test::test_tt_ij_ji_1()";

	try {

	bispace<1> si(10), sj(11);
	bispace<2> sij(si|sj), sji(sj|si);
	btensor<2> bt1(sij), bt2(sji);

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	permutation<2> p1, p2;
	p2.permute(0, 1);
	double c_ref = btod_dotprod<2>(bt1, p1, bt2, p2).calculate();

	letter i, j;
	double c = dot_product(bt1(i|j), bt2(j|i));
	check_ref(testname, c, c_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void dot_product_test::test_te_ij_ij_1() throw(libtest::test_exception) {

	static const char *testname = "dot_product_test::test_te_ij_ij_1()";

	try {

	bispace<1> si(10), sj(11);
	bispace<2> sij(si|sj);
	btensor<2> bt1(sij), bt2(sij), bt3(sij), bt4(sij);

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	btod_copy<2>(bt2).perform(bt4);
	btod_copy<2>(bt3).perform(bt4, 0.5);
	double c_ref = btod_dotprod<2>(bt1, bt4).calculate();

	letter i, j;
	double c = dot_product(bt1(i|j), bt2(i|j) + 0.5 * bt3(i|j));
	check_ref(testname, c, c_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

}


void dot_product_test::test_te_ij_ji_1() throw(libtest::test_exception) {

	static const char *testname = "dot_product_test::test_te_ij_ji_1()";

	try {

	bispace<1> si(10), sj(11);
	bispace<2> sij(si|sj), sji(sj|si);
	btensor<2> bt1(sij), bt2(sij), bt3(sji), bt4(sij);

	btod_random<2>().perform(bt1);
	btod_random<2>().perform(bt2);
	btod_random<2>().perform(bt3);
	permutation<2> perm;
	perm.permute(0, 1);
	btod_copy<2>(bt2).perform(bt4);
	btod_copy<2>(bt3, perm).perform(bt4, 0.5);
	double c_ref = btod_dotprod<2>(bt1, bt4).calculate();

	letter i, j;
	double c = dot_product(bt1(i|j), bt2(i|j) + 0.5 * bt3(j|i));
	check_ref(testname, c, c_ref);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


void dot_product_test::check_ref(const char *testname, double d, double d_ref)
	throw(libtest::test_exception) {

	if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (res), "
			<< d_ref << " (ref), " << d - d_ref << " (diff)";
		fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
	}
}



} // namespace libtensor
