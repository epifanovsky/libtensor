#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libvmm.h>
#include "compare_ref.h"
#include "tod_set_diag_test.h"

namespace libtensor {


void tod_set_diag_test::perform() throw(libtest::test_exception) {

	srand48(time(NULL));

	index<2> i2a, i2b;

	i2b[0] = 10; i2b[1] = 10;
	dimensions<2> dims2_10(index_range<2>(i2a, i2b));
	run_test(dims2_10, 0.0);
	run_test(dims2_10, 11.5);
}


template<size_t N>
void tod_set_diag_test::run_test(const dimensions<N> &dims, double d)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_set_diag_test::run_test(" << dims << ")";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	tensor<N, double, allocator_t> t(dims), t_ref(dims);
	tensor_ctrl<N, double> ctrl(t), ctrl_ref(t_ref);

	double *p = ctrl.req_dataptr();
	double *p_ref = ctrl_ref.req_dataptr();

	//	Fill in random data & prepare the reference

	abs_index<N> ai(dims);
	do {
		size_t n = ai.get_index().at(0);
		bool diag = true;
		for(size_t j = 1; j < N; j++) {
			if(ai.get_index().at(j) != n) {
				diag = false;
				break;
			}
		}
		if(diag) {
			p[ai.get_abs_index()] = drand48();
			p_ref[ai.get_abs_index()] = d;
		} else {
			p[ai.get_abs_index()] =
				p_ref[ai.get_abs_index()] = drand48();
		}
	} while(ai.inc());

	ctrl.ret_dataptr(p); p = NULL;
	ctrl_ref.ret_dataptr(p_ref); p_ref = NULL;
	t_ref.set_immutable();

	//	Run the operation

	tod_set_diag<N>(d).perform(t);

	//	Compare against the reference

	compare_ref<N>::compare(tnss.str().c_str(), t, t_ref, 0.0);

	} catch(exception &e) {
		fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
