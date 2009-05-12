#include "bispace.h"
#include "btensor.h"
#include "direct_btensor_test.h"

namespace libtensor {

void direct_btensor_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(10), a_sp(20);
	i_sp.split(5); a_sp.split(5).split(10).split(15);
	bispace<2> ia(i_sp*a_sp);
	btensor<2> bt1(ia), bt2(ia);

	letter i, a;

	direct_btensor<2> dbt(i|a, bt1(i|a) + bt2(i|a));
}

} // namespace libtensor

