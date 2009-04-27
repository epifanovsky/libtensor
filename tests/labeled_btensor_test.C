#include "bispace.h"
#include "btensor.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_test.h"

namespace libtensor {

void labeled_btensor_test::perform() throw(libtest::test_exception) {
	bispace<1> sp_i(10), sp_j(10), sp_a(20), sp_b(20);
	bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
	bispace<2> sp_ji(sp_j&sp_i), sp_ba(sp_b&sp_a);
	bispace<4> sp_ijab(sp_ij*sp_ab), sp_jiab(sp_ji*sp_ab);
	btensor<4> t1_ijab(sp_ijab), t2_ijab(sp_ijab);
	btensor<4> t3_jiab(sp_jiab), t4_jiab(sp_jiab);

	letter i, j, a, b;

	t1_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b) +
		t4_jiab(j|i|a|b);
	t1_ijab(i|j|a|b) + (t2_ijab(i|j|a|b) + t3_jiab(j|i|a|b));
	(t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)) +
		(t3_jiab(j|i|a|b) + t4_jiab(j|i|a|b));

	0.5*t1_ijab(i|j|a|b);
	t1_ijab(i|j|a|b)*0.5;
	0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b)*2.0 + t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + 0.5*t2_ijab(i|j|a|b);
	t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
	0.5*t1_ijab(i|j|a|b) + t2_ijab(i|j|a|b)*2.0;
}

} // namespace libtensor
