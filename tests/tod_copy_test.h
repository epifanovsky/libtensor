#ifndef LIBTENSOR_TOD_COPY_TEST_H
#define LIBTENSOR_TOD_COPY_TEST_H

#include <cmath>
#include <libtest.h>
#include <libvmm.h>
#include "tod_copy.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Tests the libtensor::tod_copy class

	\ingroup libtensor_tests
**/
class tod_copy_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	/**	\brief Tests if an exception is throws when the tensors have
			different dimensions
	**/
	void test_exc() throw(libtest::test_exception);

	template<size_t N>
	void test_operation(const dimensions<N> &dim)
		throw(libtest::test_exception);

};

template<size_t N>
void tod_copy_test::test_operation(const dimensions<N> &dim)
	throw(libtest::test_exception) {

	size_t sz = dim.get_size();

	typedef libvmm::std_allocator<double> allocator;
	tensor<N,double,allocator> ta(dim), tb(dim), tc(dim);
	tensor_ctrl<N,double> tca(ta), tcb(tb), tcc(tc);

	double *dref = new double[sz];
	double *pa = tca.req_dataptr();
	for(size_t i=0; i<sz; i++) pa[i]=dref[i]=drand48();
	tca.ret_dataptr(pa);
	ta.set_immutable();

	tod_copy<N> cp(ta);
	cp.perform(tb);

	const double *pb = tcb.req_const_dataptr();
	bool ok = true;
	size_t ielem;
	double dfail_ref, dfail_act;
	for(ielem=0; ielem<sz; ielem++) {
		if(fabs(pb[ielem]-dref[ielem])>fabs(dref[ielem])*5e-15) {
			dfail_ref = dref[ielem]; dfail_act = pb[ielem];
			ok=false; break;
		}
	}
	tcb.ret_dataptr(pb); pb=NULL;

	if(!ok) {
		delete [] dref;
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_copy_test::"
			"test_operation<%lu>()", N);
		snprintf(msg, 1024, "The copy (1) does not match reference at "
			"element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}

	cp.perform(tc);

	const double *pc = tcc.req_const_dataptr();
	for(ielem=0; ielem<sz; ielem++) {
		if(fabs(pc[ielem]-dref[ielem])>fabs(dref[ielem])*5e-15) {
			dfail_ref = dref[ielem]; dfail_act = pc[ielem];
			ok=false; break;
		}
	}
	tcc.ret_dataptr(pc); pc=NULL;

	if(!ok) {
		delete [] dref;
		char method[1024], msg[1024];
		snprintf(method, 1024, "tod_copy_test::"
			"test_operation<%lu>()", N);
		snprintf(msg, 1024, "The copy (2) does not match reference at "
			"element %lu: %lg(ref) vs %lg(act), "
			"%lg(diff)", ielem, dfail_ref, dfail_act,
			dfail_act-dfail_ref);
		fail_test(method, __FILE__, __LINE__, msg);
	}

	delete [] dref;
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_TEST_H

