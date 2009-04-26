#include "bispace.h"
#include "btensor_test.h"

namespace libtensor {

void btensor_test::perform() throw(libtest::test_exception) {
	index<4> i1, i2;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	index_range<4> ir(i1,i2);
	dimensions<4> dims(ir);
	btensor<4> bt(dims);

	letter i,j,k,l;
	bt(i|j|k|l);

	bispace<1> i_sp(10), a_sp(20);
	i_sp.split(5); a_sp.split(5).split(10).split(15);
	//btensor<2> bt2(i_sp*a_sp);
}

} // namespace libtensor

