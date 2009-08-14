#include <libtensor.h>
#include "libtensor_pt_suite.h"

namespace libtensor {

template<size_t N> 
dimensions<N> libtensor_pt_suite::Small<N>::dims() {
	index<N> i1, i2;
	for ( size_t i=0; i<N; i++ ) {
		i2[i]=10;
	}
	index_range<N> ir(i1,i2);
	return dimensions<N>(ir);
}

template<size_t N> 
dimensions<N> libtensor_pt_suite::Medium<N>::dims() {
	index<N> i1, i2;
	for ( size_t i=0; i<N; i++ ) {
		i2[i]=20;
	}
	index_range<N> ir(i1,i2);
	return dimensions<N>(ir);
}


template<size_t N> 
dimensions<N> libtensor_pt_suite::Large<N>::dims() {
	index<N> i1, i2;
	for ( size_t i=0; i<N; i++ ) {
		i2[i]=40;
	}
	index_range<N> ir(i1,i2);
	return dimensions<N>(ir);
}

libtensor_pt_suite::libtensor_pt_suite() 
	: performance_test_suite("libtensor_performance_test") 
{
	add_tests("tod_add (1,small)",m_tod_add_ptsc1);
	add_tests("tod_add (2,medium)",m_tod_add_ptsc2);
	add_tests("tod_add (3,large)",m_tod_add_ptsc3);
}

}

