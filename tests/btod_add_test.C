#include <libvmm.h>
#include <libtensor.h>
#include "btod_add_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef block_tensor<2,double,allocator> block_tensor2;

void btod_add_test::perform() throw(libtest::test_exception) {
	test_exc();

	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai);

	block_tensor2 bt1(ia), bt2(ai);
	permutation<2> p;

	btod_add<2> operation(p);
	p.permute(0,1);
	operation.add_op(bt2,p,0.5);
	operation.perform(bt1,0.1);

        operation.perform(bt1);
}

void btod_add_test::test_exc() throw(libtest::test_exception) {
	index<2> i1, i2; i2[0] = 1; i2[1] = 2;
	dimensions<2> dims_ia(index_range<2>(i1, i2));
	i2[0] = 2; i2[1] = 1;
	dimensions<2> dims_ai(index_range<2>(i1, i2));
	block_index_space<2> ia(dims_ia), ai(dims_ai);

	block_tensor2 bt1(ia), bt2(ai);
	permutation<2> p1,p2;
	p1.permute(0,1);

	btod_add<2> add(p1);

	bool ok=false;
	try {
		add.add_op(bt1,p2,0.5);
		add.add_op(bt2,p2,1.0);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("btod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous operands");
	}

	ok=false;
	try {
		add.add_op(bt2,p1,1.0);
		add.perform(bt1);
	}
	catch(exception e) {
		ok=true;
	}

	if(!ok) {
		fail_test("btod_add_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception due to heterogeneous result tensor");
	}

}


} // namespace libtensor

