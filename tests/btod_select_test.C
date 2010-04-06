#include <sstream>
#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/tensor.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btod/btod_select.h>
#include <libtensor/tod/tod_btconv.h>
#include <libtensor/tod/tod_select.h>
#include "btod_select_test.h"


namespace libtensor {


void btod_select_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


void btod_select_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	tod_select<2>::list_t tlist;
	btod_select<2>::list_t btlist;


	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	btod_select<2>().perform(bt, btlist, 5);
	tod_select<2>().perform(t_ref, tlist, 5);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	typename tod_select<2>::list_t::iterator it = tlist.begin();
	btod_select<2>::list_t::iterator ibt = btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if ( it->value != ibt->value ) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		index<2> idx=bis.get_block_start(ibt->bidx);
		idx[0]+=ibt->idx[0];
		idx[1]+=ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		it++; ibt++;
	}

}


void btod_select_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_select_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	index<2> i1, i2;
	i2[0] = 5; i2[1] = 8;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	mask<2> m01, m10;
	m01[1] = true; m10[0] = true;
	bis.split(m10, 3);
	bis.split(m01, 4);
	block_tensor<2, double, allocator_t> bt(bis);
	tensor<2, double, allocator_t> t_ref(dims);
	tod_select<2>::list_t tlist;
	btod_select<2>::list_t btlist;

	try {

	//	Fill in random data
	//
	btod_random<2>().perform(bt);
	tod_btconv<2>(bt).perform(t_ref);

	btod_select<2>().perform(bt, btlist, 10);
	tod_select<2>().perform(t_ref, tlist, 10);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}

	tod_select<2>::list_t::iterator it=tlist.begin();
	btod_select<2>::list_t::iterator ibt=btlist.begin();
	while (it != tlist.end() && ibt != btlist.end()) {
		if ( it->value != ibt->value ) {
			std::ostringstream oss;
			oss << "Value of list element does not match reference "
					<< "(found: " << ibt->value
					<< ", expected: " << it->value << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}
		index<2> idx=bis.get_block_start(ibt->bidx);
		idx[0]+=ibt->idx[0];
		idx[1]+=ibt->idx[1];
		if (! idx.equals(it->idx)) {
			std::ostringstream oss;
			oss << "Index of list element does not match reference "
					<< "(found: " << idx
					<< ", expected: " << it->idx << ").";
			fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
		}

		it++; ibt++;
	}
}


} // namespace libtensor
