#include "bispace.h"
#include "btensor_test.h"

namespace libtensor {

void btensor_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(10), a_sp(20);
	i_sp.split(5); a_sp.split(5).split(10).split(15);
	bispace<2> ia(i_sp*a_sp);
	btensor<2> bt2(ia);

	dimensions<2> bt2_dims(bt2.get_dims());
	if(bt2_dims[0] != 10) {
		fail_test("btensor_test::perform()", __FILE__, __LINE__,
			"Block tensor bt2 has the wrong dimension: i");
	}

	if(bt2_dims[1] != 20) {
		fail_test("btensor_test::perform()", __FILE__, __LINE__,
			"Block tensor bt2 has the wrong dimension: a");
	}

//	letter i,j,k,l;
//	bt(i|j|k|l);
}

} // namespace libtensor

