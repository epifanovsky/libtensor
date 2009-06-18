#include <libtensor.h>
#include "contract_test.h"

namespace libtensor {

void contract_test::perform() throw(libtest::test_exception) {
	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij*sp_ab), sp_jiab(sp_ji*sp_ab);
	bispace<4> sp_abcd(sp_ab&sp_ab);
	btensor<4> t1_ijab(sp_ijab), t2_ijab(sp_ijab);
	btensor<4> t3_jiab(sp_jiab), t4_jiab(sp_jiab);
	btensor<4> t5_abcd(sp_abcd), t6_abcd(sp_abcd), t7_abcd(sp_abcd);

	letter i, j, k, l, a, b, c, d;
/*
	contract(i, t1_ijab(i|j|a|b), t2_ijab(i|k|c|d));
	t5_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d));
	t6_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b),
		t2_ijab(i|j|c|d));
	t7_abcd(a|b|c|d) = t5_abcd(a|b|c|d) + t6_abcd(a|b|c|d);
	t7_abcd(a|b|c|d) = contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d)) +
		contract(i|j, t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b),
			t2_ijab(i|j|c|d)) + t6_abcd(a|b|c|d);
	contract(i|j, t1_ijab(i|j|a|b), t2_ijab(i|j|c|d) + t3_jiab(j|i|c|d));
	*/
}

} // namespace libtensor
