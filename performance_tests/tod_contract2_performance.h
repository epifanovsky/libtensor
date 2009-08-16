#ifndef TOD_CONTRACT2_PERFORMANCE_H
#define TOD_CONTRACT2_PERFORMANCE_H


#include <libtest.h>
#include <libtensor.h>

#include "performance_test.h"
#include "../timings.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Reference for performance tests of libtensor::tod_contract2 class
 
 	\param Repeats number of repeats
 	\param N order of the first %tensor less contraction order 
 	\param M order of the second %tensor less contraction order
 	\param K contraction order
 	\param X information about the size of the tensors 

 	Tests performance of
 	\f[ A += 0.5 B * C' \f]
 	where the \f$ i_k \f$ refer to the k-th index group 

 	Don't use with N<2    

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, size_t M, size_t K, typename X>
class tod_contract2_ref 
	: public performance_test<Repeats>,
	  public timings<tod_contract2_ref<Repeats,N,M,K,X> > 
{
	friend timings<tod_contract2_ref<Repeats,N,M,K,X> >; 
	static const char* k_clazz;
protected:
	virtual void do_calculate();
};

/**	\brief First performance test of the libtensor::tod_contract2 class
  
 	Tests performance of
 	\f[ A_{i_1,i_2} += 0.5 \sum_{i_3} B_{i_1,i_3} C_{i_2,i_3} \f]
 	where the \f$ i_k \f$ refer to the k-th index group 


	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, size_t M, size_t K, typename X> 
class tod_contract2_p1
	: public performance_test<Repeats> 
{
protected:
	virtual void do_calculate(); 
};

/**	\brief Second performance test of the libtensor::tod_contract2 class
  
 	Tests performance of
 	\f[ A_{i_1,i_2} += 0.5 \sum_{i_3} B_{i_1,i_3} C_{i_3,i_2} \f]
 	where the \f$ i_k \f$ refer to the k-th index group.

 	Don't use with N<2    

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, size_t M, size_t K, typename X> 
class tod_contract2_p2
	: public performance_test<Repeats> 
{
protected:
	virtual void do_calculate(); 
};

/**	\brief Third performance test of the libtensor::tod_contract2 class
  
 	Tests performance of
 	\f[ 
 	A_{i_1,i_2} += 0.5 \mathcal{P}_A \sum_{i_3} B_{i_1,i_3} C_{i_2,i_3} 
 	\f]
 	where the \f$ i_k \f$ refer to the k-th index group and 
 	\f$ \mathcal{P}_A \f$ to an arbitrary permutation    

 	Don't use with N<2    

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, size_t M, size_t K, typename X> 
class tod_contract2_p3
	: public performance_test<Repeats> 
{
protected:
	virtual void do_calculate(); 
};

/**	\brief Forth performance test of the libtensor::tod_contract2 class
  
 	Tests performance of
 	\f[ 
 	A_{i_1,i_2} += 0.5 \sum_{i_3} B_{i_1,i_3} \mathcal{P}_C C_{i_2,i_3} 
 	\f]
 	where the \f$i_k\f$ refer to the k-th index group and 
 	\f$\mathcal{P}_C\f$ to a permutation that only permutes indices of index 
 	group i_3.
 	
 	Don't use with N<3    

	\ingroup libtensor_tests
**/
template<size_t Repeats, size_t N, size_t M, size_t K, typename X> 
class tod_contract2_p4
	: public performance_test<Repeats> 
{
protected:
	virtual void do_calculate(); 
};


template<size_t R, size_t N, size_t M, size_t K, typename X>
const char* tod_contract2_ref<R,N,M,K,X>::k_clazz="tod_contract2_ref<R,N,M,K,X>";

template<size_t R, size_t N, size_t M, size_t K, typename X>
void tod_contract2_ref<R,N,M,K,X>::do_calculate() 
{
	X d;
	dimensions<N+M> dima(d.dimA());
	dimensions<N+K> dimb(d.dimB());
	dimensions<M+K> dimc(d.dimC());
	size_t sizeN=1, sizeM=1, sizeK=1;
	for ( unsigned int i=0; i<N; i++ )   sizeN*=dima.get_dim(i);
	for ( unsigned int i=N; i<N+M; i++ ) sizeM*=dima.get_dim(i);
	for ( unsigned int i=N; i<N+K; i++ ) sizeK*=dimb.get_dim(i);
	
	double* ptra=new double[sizeN*sizeM];
	double* ptrb=new double[sizeN*sizeK];
	double* ptrc=new double[sizeM*sizeK];
	for ( size_t i=0; i<sizeN*sizeM; i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<sizeN*sizeK; i++ ) ptrb[i]=drand48();
	for ( size_t i=0; i<sizeM*sizeK; i++ ) ptrc[i]=drand48();
	
	timings<tod_contract2_ref<R,N,M,K,X> >::start_timer();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, sizeN, sizeM, sizeK, 
				0.5, ptrb, sizeK, ptrc, sizeK, 1.0, ptra, sizeM);	   	
	timings<tod_contract2_ref<R,N,M,K,X> >::stop_timer();
	
	delete [] ptra;
	delete [] ptrb;
	delete [] ptrc;
}			

template<size_t R, size_t N, size_t M, size_t K, typename X>
void tod_contract2_p1<R,N,M,K,X>::do_calculate() 
{
	X d;	
	
	dimensions<N+M> dima(d.dimA());
	dimensions<N+K> dimb(d.dimB());
	dimensions<M+K> dimc(d.dimC());
	tensor<N+M, double, libvmm::std_allocator<double> > ta(dima);
	tensor<N+K, double, libvmm::std_allocator<double> > tb(dimb);
	tensor<M+K, double, libvmm::std_allocator<double> > tc(dimc);
	tensor_ctrl<N+M,double> tca(ta); 
	tensor_ctrl<N+K,double> tcb(tb); 
	tensor_ctrl<M+K,double> tcc(tc); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	double *ptrc=tcc.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	for ( size_t i=0; i<dimc.get_size(); i++ ) ptrc[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	tcc.ret_dataptr(ptrc);
	
	// start tod_contract2 calculation
	permutation<N+M> perm;
	contraction2<N,M,K> contr(perm);
	for (size_t i=0; i<K; i++) contr.contract(i+N,i+M);
	tod_contract2<N,M,K> op(contr,tb,tc);
	op.perform(ta,0.5);
}

template<size_t R, size_t N, size_t M, size_t K, typename X>
void tod_contract2_p2<R,N,M,K,X>::do_calculate() 
{
	X d;	
	dimensions<N+M> dima(d.dimA());
	dimensions<N+K> dimb(d.dimB());
	dimensions<M+K> dimc(d.dimC());
	
	char a[M+K], b[M+K];		
	for (size_t i=0; i<K; i++) {
		a[i]=i; b[i]=i+M;  
	}
	for (size_t i=K; i<M+K; i++) {
		a[i]=i; b[i]=i-K;  
	}
	permutation_builder<M+K> pbc(a,b);
	dimc.permute(pbc.get_perm());
	
	tensor<N+M, double, libvmm::std_allocator<double> > ta(dima);
	tensor<N+K, double, libvmm::std_allocator<double> > tb(dimb);
	tensor<M+K, double, libvmm::std_allocator<double> > tc(dimc);
	tensor_ctrl<N+M,double> tca(ta); 
	tensor_ctrl<N+K,double> tcb(tb); 
	tensor_ctrl<M+K,double> tcc(tc); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	double *ptrc=tcc.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	for ( size_t i=0; i<dimc.get_size(); i++ ) ptrc[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	tcc.ret_dataptr(ptrc);
	
	// start tod_contract2 calculation
	permutation<N+M> perm;
	contraction2<N,M,K> contr(perm);
	for (size_t i=0; i<K; i++) contr.contract(i+N,i);
	tod_contract2<N,M,K> op(contr,tb,tc);
	op.perform(ta,0.5);
}


template<size_t R, size_t N, size_t M, size_t K, typename X>
void tod_contract2_p3<R,N,M,K,X>::do_calculate() 
{
	X d;	

	dimensions<N+M> dima(d.dimA());
	dimensions<N+K> dimb(d.dimB());
	dimensions<M+K> dimc(d.dimC());
	permutation<N+M> perma;
	for (size_t i=0; i<((N<M)?N:M); i++) {
		perma.permute(i,N+M-1-i);
	}
	dima.permute(perma);
	
	tensor<N+M, double, libvmm::std_allocator<double> > ta(dima);
	tensor<N+K, double, libvmm::std_allocator<double> > tb(dimb);
	tensor<M+K, double, libvmm::std_allocator<double> > tc(dimc);
	tensor_ctrl<N+M,double> tca(ta); 
	tensor_ctrl<N+K,double> tcb(tb);  
	tensor_ctrl<M+K,double> tcc(tc); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	double *ptrc=tcc.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	for ( size_t i=0; i<dimc.get_size(); i++ ) ptrc[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	tcc.ret_dataptr(ptrc);
	
	// start tod_contract2 calculation
	contraction2<N,M,K> contr(perma);
	for (size_t i=0; i<K; i++) contr.contract(i+N,i+M);
	tod_contract2<N,M,K> op(contr,tb,tc);
	op.perform(ta,1.0);
}


template<size_t R, size_t N, size_t M, size_t K, typename X>
void tod_contract2_p4<R,N,M,K,X>::do_calculate() 
{
	X d;	
	dimensions<N+M> dima(d.dimA());
	dimensions<N+K> dimb(d.dimB());
	dimensions<M+K> dimc(d.dimC());

	permutation<M+K> permc;
	for (size_t i=0; i<K/2; i++) 
		permc.permute(M+i,M+K-1-i);
	dimc.permute(permc);
	
	tensor<N+M, double, libvmm::std_allocator<double> > ta(dima);
	tensor<N+K, double, libvmm::std_allocator<double> > tb(dimb);
	tensor<M+K, double, libvmm::std_allocator<double> > tc(dimc);
	tensor_ctrl<N+M,double> tca(ta); 
	tensor_ctrl<N+K,double> tcb(tb); 
	tensor_ctrl<N+K,double> tcc(tc); 
	
	double *ptra=tca.req_dataptr();
	double *ptrb=tcb.req_dataptr();
	double *ptrc=tcc.req_dataptr();
	for ( size_t i=0; i<dima.get_size(); i++ ) ptra[i]=drand48();
	for ( size_t i=0; i<dimb.get_size(); i++ ) ptrb[i]=drand48();
	for ( size_t i=0; i<dimc.get_size(); i++ ) ptrc[i]=drand48();
	tca.ret_dataptr(ptra);
	tcb.ret_dataptr(ptrb);
	tcc.ret_dataptr(ptrc);
	
	// start tod_contract2 calculation
	permutation<N+M> perm;
	contraction2<N,M,K> contr(perm);
	for (size_t i=0; i<K; i++) contr.contract(M+i,M+K-1-i);
	tod_contract2<N,M,K> op(contr,tb,tc);
	op.perform(ta,1.0);
}


} // namespace libtensor

#endif // TOD_CONTRACT2_PERFORMANCE_H

