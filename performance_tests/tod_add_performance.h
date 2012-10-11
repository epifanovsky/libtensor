#ifndef TOD_ADD_PERFORMANCE_H
#define TOD_ADD_PERFORMANCE_H


#include <libtest/libtest.h>
#include <libtensor/core/allocator.h>
#include <libtensor/libtensor.h>
#include <libtensor/linalg/linalg.h>
#include "performance_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Reference for performance tests of libtensor::tod_add class

 	\param Repeats number of repeats
 	\param DimData information about the size of the tensors

 	Tests performance of
 	\f[ A = A + 2.0 B \f]
 	The size of A and B is determined by function dimA() of the DimData object.

	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, typename DimData>
class tod_add_ref
	: public performance_test<Repeats>,
	  public timings<tod_add_ref<Repeats,DimData> >
{
	friend class timings<tod_add_ref<Repeats,DimData> >;
public:
	static const char* k_clazz;
protected:
	virtual void do_calculate();
};


/**	\brief First performance test of the libtensor::tod_add class

 	Tests performance of
 	\f[ A = A + 2.0 B \f]
	The size of A and B is determined by function dimA() of the DimData object.

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, size_t N, typename DimData>
class tod_add_p1
	: public performance_test<Repeats>
{
protected:
	virtual void do_calculate();
};

/**	\brief Second performance test of the libtensor::tod_add class

 	Tests performance of
 	\f[ A = A + 2.0 \mathcal{P}_B B \f]
 	where \f$ \mathcal{P}_B \f$ refers to a permutation which inverts the
 	sequence of the indices, e.g. (0123)->(3210).

	The size of A and B is determined by function dimA() of the DimData object.

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, size_t N, typename DimData>
class tod_add_p2
	: public performance_test<Repeats>
{
protected:
	virtual void do_calculate();
};

/**	\brief Third performance test of the libtensor::tod_add class

 	Tests performance of
 	\f[ A = A + 2.0 \mathcal{P}_B B \f]
 	where \f$ \mathcal{P}_B \f$ refers to a permutation which changes the
 	sequence of groups of indices, e.g. (0123)->(2301)

	The size of A and B is determined by function dimA() of the DimData object.

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats, size_t N, typename DimData>
class tod_add_p3
	: public performance_test<Repeats>
{
protected:
	virtual void do_calculate();
};


template<size_t R, typename DimData>
const char* tod_add_ref<R,DimData>::k_clazz="tod_add_ref<R,DimData>";

template<size_t R, typename DimData>
void tod_add_ref<R,DimData>::do_calculate()
{
	DimData d;
	size_t total_size=d.dimA().get_size();

	double* ptra=new double[total_size];
	double* ptrb=new double[total_size];
	for ( size_t i=0; i<total_size; i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<total_size; i++ ) ptrb[i]=drand48();

	timings<tod_add_ref<R,DimData> >::start_timer();
	linalg::mul2_i_i_x(0, total_size, ptrb, 1, 2.0, ptra, 1);
	timings<tod_add_ref<R,DimData> >::stop_timer();

	delete [] ptra;
	delete [] ptrb;
}

template<size_t R, size_t N, typename DimData>
void tod_add_p1<R,N,DimData>::do_calculate()
{
	DimData d;
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
	permutation<N> perm;
	tod_add<N> add(tb,2.0);
	add.prefetch();
	add.perform(false, ta);
}

template<size_t R, size_t N, typename DimData>
void tod_add_p2<R,N,DimData>::do_calculate()
{
	DimData d;
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
	tod_add<N> add(tb,permb,2.0);
	add.prefetch();
	add.perform(false, ta);
}


template<size_t R, size_t N, typename DimData>
void tod_add_p3<R,N,DimData>::do_calculate()
{
	DimData d;
	dimensions<N> dima(d.dimA()), dimb(d.dimA());
	permutation<N> permb;
	for ( size_t i=0; i<N/2; i++ )
		permb.permute(i,i+(N+1)/2);

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
	tod_add<N> add(tb,permb,2.0);
	add.prefetch();
	add.perform(false, ta);
}


} // namespace libtensor

#endif // TOD_ADD_PERFORMANCE_H

