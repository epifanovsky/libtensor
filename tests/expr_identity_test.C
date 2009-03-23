#include <libvmm.h>
#include "tensor.h"
#include "expr_identity_test.h"

namespace libtensor {

typedef tensor< 2,double,libvmm::std_allocator<double> > tensor2_d;

void expr_identity_test::perform() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0]=10; i2[1]=10;
	index_range<2> ir(i1,i2);
	dimensions<2> dims(ir);
	tensor2_d t(dims);

	expr_identity<tensor2_d> expr1(t);
	expr_identity<tensor2_d> expr2(expr1);
}

} // namespace libtensor

