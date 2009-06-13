#include <libtensor.h>
#include "tod_symcontract2_test.h"

namespace libtensor {

typedef tensor<2, double, libvmm::std_allocator<double> > tensor2_d;
typedef tensor<4, double, libvmm::std_allocator<double> > tensor4_d;

void tod_symcontract2_test::perform() throw(libtest::test_exception) {
	// since we know that tod_contract2, tod_add and tod_copy work 
	// correctly, only the following tests have to be performed:
	//  1. is the result R of tod_symcontract symmetric?
	//  2. do tod_contract2 and tod_add yield the same result?
	index<4> i1, i2;
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	tensor4_d t(dim);

			
}

} // namespace libtensor

