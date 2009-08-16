#include <libtensor.h>
#include "libtensor_pt_suite.h"

namespace libtensor {

template<size_t N, size_t M, size_t K> 
dimensions<N+M> 
libtensor_pt_suite::dimensions_t<N,M,K>::dimA() 
{
	index_range<N+M> ir(index<N+M>(),ind1);
	return dimensions<N+M>(ir);
}

template<size_t N, size_t M, size_t K> 
dimensions<N+K> 
libtensor_pt_suite::dimensions_t<N,M,K>::dimB() 
{
	index_range<N+K> ir(index<N+K>(),ind2);
	return dimensions<N+K>(ir);
}

template<size_t N, size_t M, size_t K> 
dimensions<M+K> 
libtensor_pt_suite::dimensions_t<N,M,K>::dimC() 
{
	index_range<M+K> ir(index<M+K>(),ind3);
	return dimensions<M+K>(ir);
}

template<size_t N, size_t M, size_t K> 
libtensor_pt_suite::small_t<N,M,K>::small_t()
{ 
	for ( size_t i=0; i<N; i++ ) {
		small_t::ind1[i]=10+i;
		small_t::ind2[i]=small_t::ind1[i];
	}
	
	for ( size_t i=0; i<M; i++ ) {
		small_t::ind3[i]=10+N+i;
		small_t::ind1[i+N]=small_t::ind3[i];
	}
	
	for ( size_t i=0; i<K; i++ ) {
		small_t::ind2[i+N]=10+N+M+i;
		small_t::ind3[i+M]=small_t::ind2[i+N];
	}
}

template<size_t N, size_t M, size_t K> 
libtensor_pt_suite::medium_t<N,M,K>::medium_t()
{ 
	for ( size_t i=0; i<N; i++ ) {
		medium_t::ind1[i]=20+i;
		medium_t::ind2[i]=medium_t::ind1[i];
	}
	for ( size_t i=0; i<M; i++ ) {
		medium_t::ind3[i]=20+N+i;
		medium_t::ind1[i+N]=medium_t::ind3[i];
	}
	
	for ( size_t i=0; i<K; i++ ) {
		medium_t::ind2[i+N]=20+N+M+i;
		medium_t::ind3[i+M]=medium_t::ind2[i+N];
	}
}
template<size_t N, size_t M, size_t K> 
libtensor_pt_suite::large_t<N,M,K>::large_t()
{ 
	for ( size_t i=0; i<N; i++ ) {
		large_t::ind1[i]=30+i;
		large_t::ind2[i]=large_t::ind1[i]; 
	}
	for ( size_t i=0; i<M; i++ ) {
		large_t::ind3[i]=30+N+i;
		large_t::ind1[i+N]=large_t::ind3[i];
	}
	
	for ( size_t i=0; i<K; i++ ) {
		large_t::ind2[i+N]=30+N+M+i;
		large_t::ind3[i+M]=large_t::ind2[i+N];
	}
}

libtensor_pt_suite::libtensor_pt_suite() 
	: performance_test_suite("libtensor_performance_test") 
{
	add_tests("tod_add (1,small)",m_tod_add_ptsc1);
	add_tests("tod_add (2,medium)",m_tod_add_ptsc2);
	add_tests("tod_add (3,large)",m_tod_add_ptsc3);
	add_tests("tod_contract2 (1,small)",m_tod_contract2_ptsc1);
	add_tests("tod_contract2 (2,medium)",m_tod_contract2_ptsc2);
	add_tests("tod_contract2 (3,large)",m_tod_contract2_ptsc3);
	add_tests("tod_copy (1,small)",m_tod_copy_ptsc1);
	add_tests("tod_copy (2,medium)",m_tod_copy_ptsc2);
	add_tests("tod_copy (3,large)",m_tod_copy_ptsc3);
	add_tests("tod_dotprod (1,small)",m_tod_dotprod_ptsc1);
	add_tests("tod_dotprod (2,medium)",m_tod_dotprod_ptsc2);
	add_tests("tod_dotprod (3,large)",m_tod_dotprod_ptsc3);
}

}

