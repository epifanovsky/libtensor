#include <sstream>
#include <libvmm.h>
#include "tod_copy_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef tensor<4, double, allocator> tensor4;
typedef tensor_ctrl<4,double> tensor4_ctrl;

void tod_copy_test::perform() throw(libtest::test_exception) {
	test_exc();

	index<2> i2a, i2b; i2b[0]=10; i2b[1]=12;
	index_range<2> ir2(i2a, i2b); dimensions<2> dims2(ir2);
	test_plain(dims2);
}

template<size_t N>
void tod_copy_test::test_plain(const dimensions<N> &dims)
	throw(libtest::test_exception) {

	tensor<N, double, allocator> ta(dims), tb(dims), tb_ref(dims);
	tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

	double *dta = tca.req_dataptr();
	double *dtb1 = tcb.req_dataptr();
	double *dtb2 = tcb_ref.req_dataptr();

	// Fill in random data

	index<N> ida;
	size_t sz = 0;
	do {
		size_t i;
		i = dims.abs_index(ida);
		dta[i] = dtb2[i] = drand48();
		dtb1[i] = drand48();
		sz++;
	} while(dims.inc_index(ida));
	printf(" sz=%lu ", sz);
	tca.ret_dataptr(dta); dta = NULL;
	tcb.ret_dataptr(dtb1); dtb1 = NULL;
	tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
	ta.set_immutable(); tb_ref.set_immutable();

	// Invoke the copy operation

	tod_copy<N> cp(ta);
	cp.perform(tb);

	// Compare against the reference

	compare_ref("tod_copy_test::test_plain()", tb, tb_ref, 0.0);
}

void tod_copy_test::test_exc() throw(libtest::test_exception) {
	index<4> i1, i2, i3;
	i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
	i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
	index_range<4> ir1(i1,i2), ir2(i1,i3);
	dimensions<4> dim1(ir1), dim2(ir2);
	tensor4 t1(dim1), t2(dim2);

	bool ok = false;
	try {
		tod_copy<4> tc(t1); tc.perform(t2);
	} catch(exception e) {
		ok = true;
	}

	if(!ok) {
		fail_test("tod_copy_test::test_exc()", __FILE__, __LINE__,
			"Expected an exception with heterogeneous arguments");
	}
}

template<size_t N>
void tod_copy_test::compare_ref(const char *test,
	tensor_i<N, double> &t, tensor_i<N, double> &t_ref, double thresh)
	throw(libtest::test_exception) {

	tod_compare<N> cmp(t, t_ref, thresh);
	if(!cmp.compare()) {
		std::ostringstream ss1, ss2;
		ss2 << "Result does not match reference at element "
			<< cmp.get_diff_index() << ": "
			<< cmp.get_diff_elem_1() << " (act) vs. "
			<< cmp.get_diff_elem_2() << " (ref), "
			<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
			<< " (diff) in " << test;
		fail_test("tod_copy_test::compare_ref()",
			__FILE__, __LINE__, ss2.str().c_str());
	}

}

} // namespace libtensor

