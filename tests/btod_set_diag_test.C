#include "btod_set_diag_test.h"
#include "compare_ref.h"

namespace libtensor {


void btod_set_diag_test::perform() throw(libtest::test_exception) {

	index<2> i2a, i2b;
	mask<2> m2;

	m2[0] = true; m2[1] = true;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<2> dims2_10(index_range<2>(i2a, i2b));
	block_index_space<2> bis2_10(dims2_10);

	run_test(bis2_10, 0.0);
	run_test(bis2_10, 11.5);

	bis2_10.split(m2, 3);

	run_test(bis2_10, 0.0);
	run_test(bis2_10, 11.6);

	bis2_10.split(m2, 8);

	run_test(bis2_10, 0.0);
	run_test(bis2_10, 11.7);
}


template<size_t N>
void btod_set_diag_test::run_test(const block_index_space<N> &bis, double d)
	throw(libtest::test_exception) {

	dimensions<N> bidims(bis.get_block_index_dims());
	std::ostringstream tnss;
	tnss << "btod_set_diag_test::run_test(" << bidims << ", " << d << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	block_tensor<N, double, allocator_t> bt(bis);
	tensor<N, double, allocator_t> t(bis.get_dims()), t_ref(bis.get_dims());

	//	Fill in random data & make reference

	btod_random<N>().perform(bt);
	tod_btconv<N>(bt).perform(t_ref);
	tod_set_diag<N>(d).perform(t_ref);

	//	Perform the operation

	btod_set_diag<N>(d).perform(bt);
	tod_btconv<N>(bt).perform(t);

	//	Compare against the reference

	compare_ref<N>::compare(tnss.str().c_str(), t, t_ref, 0.0);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
