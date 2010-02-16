#include <cmath>
#include <ctime>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_diag.h>
#include "tod_diag_test.h"
#include "compare_ref.h"

namespace libtensor {


typedef libvmm::std_allocator<double> allocator;


void tod_diag_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
}


void tod_diag_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_diag_test::test_1()";

	try {

	index<1> i1a, i1b;
	i1b[0] = 10;
	index<2> i2a, i2b;
	i2b[0] = 10; i2b[1] = 10;
	dimensions<1> dims1(index_range<1>(i1a, i1b));
	dimensions<2> dims2(index_range<2>(i2a, i2b));
	size_t sza = dims2.get_size(), szb = dims1.get_size();

	tensor<2, double, allocator> ta(dims2);
	tensor<1, double, allocator> tb(dims1), tb_ref(dims1);
	tensor_ctrl<2, double> tca(ta);
	tensor_ctrl<1, double> tcb(tb), tcb_ref(tb_ref);

	double *pa = tca.req_dataptr();
	double *pb = tcb.req_dataptr();
	double *pb_ref = tcb_ref.req_dataptr();

	for(size_t i = 0; i < sza; i++) pa[i] = drand48();
	for(size_t i = 0; i < szb; i++) pb[i] = drand48();

	for(size_t i = 0; i < szb; i++) {
		index<2> idxa; idxa[0] = i; idxa[1] = i;
		index<1> idxb; idxb[0] = i;
		abs_index<2> aidxa(idxa, dims2);
		abs_index<1> aidxb(idxb, dims1);
		pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
	}

	tca.ret_dataptr(pa); pa = 0;
	tcb.ret_dataptr(pb); pb = 0;
	tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;

	mask<2> m; m[0] = true; m[1] = true;
	tod_diag<2, 2>(ta, m).perform(tb);

	compare_ref<1>::compare(testname, tb, tb_ref, 1e-15);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
