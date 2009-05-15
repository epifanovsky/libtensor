#include "btod_contract2_test.h"
#include <libvmm.h>
#include "block_tensor.h"
#include "bispace.h"
#include "contraction2.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;


void btod_contract2_test::perform() throw(libtest::test_exception) {
	bispace<1> i_sp(2), a_sp(3);
	bispace<2> ia(i_sp*a_sp), ai(a_sp*i_sp), ij(i_sp*i_sp);

	block_tensor2 bt1(ia), bt2(ai), bt3(ij);
	permutation<2> p;
	
	contraction2<1,1,1> contr(p);
	contr.contract(1,0);

	btod_contract2<1,1,1> operation(contr,bt1,bt2);
	operation.perform(bt3,0.1);

        operation.perform(bt3);

}

} // namespace libtensor

