#include <libtensor.h>
#include "labeled_btensor_test.h"

namespace libtensor {

void labeled_btensor_test::perform() throw(libtest::test_exception) {
	test_label();
	test_expr();
}

void labeled_btensor_test::test_label() throw(libtest::test_exception) {
	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<4> sp_ijab((sp_i&sp_j)*(sp_a&sp_b));
	btensor<4> t(sp_ijab);

	letter i, j, k, a, b, c;

	if(!t(i|j|a|b).contains(i)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(i)");
	}
	if(!t(i|j|a|b).contains(j)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(j)");
	}
	if(!t(i|j|a|b).contains(a)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(a)");
	}
	if(!t(i|j|a|b).contains(b)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(b)");
	}
	if(t(i|j|a|b).contains(k)) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).contains(k)");
	}

	if(t(i|j|a|b).index_of(i) != 0) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(i)");
	}
	if(t(i|j|a|b).index_of(j) != 1) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(j)");
	}
	if(t(i|j|a|b).index_of(a) != 2) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(a)");
	}
	if(t(i|j|a|b).index_of(b) != 3) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).index_of(b)");
	}

	if(t(i|j|a|b).letter_at(0) != i) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(0)");
	}
	if(t(i|j|a|b).letter_at(1) != j) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(1)");
	}
	if(t(i|j|a|b).letter_at(2) != a) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(2)");
	}
	if(t(i|j|a|b).letter_at(3) != b) {
		fail_test("labeled_btensor_test::test_label()", __FILE__,
			__LINE__, "Failed label test: t(i|j|a|b).letter_at(3)");
	}
}

void labeled_btensor_test::test_expr() throw(libtest::test_exception) {
	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij*sp_ab), sp_jiab(sp_ji*sp_ab);
	btensor<4> t1_ijab(sp_ijab), t2_ijab(sp_ijab);
	btensor<4> t3_jiab(sp_jiab), t4_jiab(sp_jiab);

	letter i, j, a, b;

	t1_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t2_ijab(i|j|a|b) = t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) +
		t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
	(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
		(t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

	t3_jiab(j|i|a|b) = 0.5*t1_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) = t1_ijab(i|j|a|b)*0.5;
	t3_jiab(j|i|a|b) = 0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b)*2.0 + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + 0.5*t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
	0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + 0.5*t3_jiab(j|i|a|b);
	0.5*t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
	0.5*t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + 2.0*t3_jiab(j|i|a|b));
	0.5*(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b));
	(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b))*0.5;
	0.5*(t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b));
	t4_jiab(j|i|a|b) = (t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b))*0.5;
	2.0*(t1_ijab(i|j|a|b) + 2.0*t2_ijab(i|j|a|b))*0.5;
}

} // namespace libtensor
