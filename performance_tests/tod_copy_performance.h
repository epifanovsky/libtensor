#ifndef TOD_COPY_PERFORMANCE_H
#define TOD_COPY_PERFORMANCE_H


#include <libtest/libtest.h>
#include <libtensor/core/allocator.h>
#include <libtensor/libtensor.h>
#include <libtensor/linalg/linalg.h>
#include "performance_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Reference for performance tests of libtensor::tod_copy class

 	\param Repeats number of repeats
 	\param X information about the size of the tensors

 	Tests performance of
 	\f[ A = 2.0 B \f]

	The size of A and B is determined by function dimA() of the X object.

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, typename X>
class tod_copy_ref
	: public performance_test<Repeats>,
	  public timings<tod_copy_ref<Repeats,X> >
{
	friend class timings<tod_copy_ref<Repeats,X> >;
public:
	static const char* k_clazz;
protected:
	virtual void do_calculate();
};


/**	\brief First performance test of the libtensor::tod_copy class

 	Tests performance of
 	\f[ A = 2.0 B \f]

	The size of A and B is determined by function dimA() of the X object.

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, typename X>
class tod_copy_p1
	: public performance_test<Repeats>
{
protected:
	virtual void do_calculate();
};

/**	\brief Second performance test of the libtensor::tod_copy class

 	Tests performance of
 	\f[ A = 2.0 \mathcal{P}_B B \f]

	The size of A and B is determined by function dimA() of the X object.

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, typename X>
class tod_copy_p2
	: public performance_test<Repeats>
{
protected:
	virtual void do_calculate();
};


template<size_t R, typename X>
const char* tod_copy_ref<R,X>::k_clazz="tod_copy_ref<R,X>";

template<size_t R, typename X>
void tod_copy_ref<R,X>::do_calculate()
{
	X d;
	size_t total_size=d.dimA().get_size();

	double* ptra=new double[total_size];
	double* ptrb=new double[total_size];
	for ( size_t i=0; i<total_size; i++ ) ptrb[i]=drand48();

	timings<tod_copy_ref<R,X> >::start_timer();
	linalg::i_i(total_size, ptrb, 1, ptra, 1);
	linalg::i_x(total_size, 2.0, ptra,1);
	timings<tod_copy_ref<R,X> >::stop_timer();

	delete [] ptra;
	delete [] ptrb;
}

template<size_t R, size_t N, typename X>
void tod_copy_p1<R,N,X>::do_calculate()
{
	X d;
	dimensions<N> dim(d.dimA());
	dense_tensor<N, double, std_allocator<double> > ta(dim), tb(dim);
	dense_tensor_ctrl<N,double> tca(ta), tcb(tb);

	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dim.get_size(); i++ ) ptrb[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);

	// start tod_add calculation
	tod_copy<N>(tb,2.0).perform(true, 1.0, ta);
}

template<size_t R, size_t N, typename X>
void tod_copy_p2<R,N,X>::do_calculate()
{
	X d;
	dimensions<N> dima(d.dimA()), dimb(d.dimA());
	permutation<N> permb;
	for ( size_t i=0; i<N/2; i++ )
		permb.permute(i,N-(i+1));

	dimb.permute(permb);

	dense_tensor<N, double, std_allocator<double> > ta(dima), tb(dimb);
	dense_tensor_ctrl<N,double> tca(ta), tcb(tb);

	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);

	// start tod_add calculation
	tod_copy<N>(tb,permb,2.0).perform(true, 1.0, ta);
}


} // namespace libtensor

#endif // TOD_ADD_P1_H
