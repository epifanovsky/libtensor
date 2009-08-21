#ifndef LIBTENSOR_PT_SUITE_H
#define LIBTENSOR_PT_SUITE_H

#include <libtest.h>
#include <libtensor.h>
#include "performance_test_suite.h"

#include "tod_add_scenario.h"
#include "tod_copy_scenario.h"
#include "tod_contract2_scenario.h"
#include "tod_dotprod_scenario.h"

namespace libtensor {


/**
	\brief Performance test suite for the tensor library (libtensor)

	This suite runs the following performance test scenarios:
	\li libtensor::tod_add_scenario
	\li libtensor::tod_contract2_scenario
	\li libtensor::tod_copy_scenario
	\li libtensor::tod_dotprod_scenario
	
**/
class libtensor_pt_suite : public performance_test_suite {
	template<size_t N, size_t M, size_t K> 
	class dimensions_t {
	protected:
		index<N+M> ind1;
		index<N+K> ind2;
		index<M+K> ind3;
	public:
		dimensions<N+M> dimA();
		dimensions<N+K> dimB();
		dimensions<M+K> dimC();
	};
	
	template<size_t N, size_t M, size_t K> 
	class small_t : public dimensions_t<N,M,K> {
	public:
		small_t();		
	};
	template<size_t N, size_t M, size_t K> 
	class medium_t : public dimensions_t<N,M,K> {
	public:
		medium_t();		
	};
	template<size_t N, size_t M, size_t K> 
	class large_t : public dimensions_t<N,M,K> {
	public:
		large_t();		
	};

	tod_add_scenario<20,4,small_t<2,2,2> > m_tod_add_ptsc1;
	tod_add_scenario<10,4,medium_t<2,2,2> > m_tod_add_ptsc2;
	tod_add_scenario<5,4,large_t<2,2,2> > m_tod_add_ptsc3;

	tod_contract2_scenario<20,2,2,2,small_t<2,2,2> > m_tod_contract2_ptsc1;
	tod_contract2_scenario<10,2,2,2,medium_t<2,2,2> > m_tod_contract2_ptsc2;
	tod_contract2_scenario<5,2,2,2,large_t<2,2,2> > m_tod_contract2_ptsc3;

	tod_copy_scenario<20,4,small_t<2,2,2> > m_tod_copy_ptsc1;
	tod_copy_scenario<10,4,medium_t<2,2,2> > m_tod_copy_ptsc2;
	tod_copy_scenario<5,4,large_t<2,2,2> > m_tod_copy_ptsc3;

	tod_dotprod_scenario<20,4,small_t<2,2,2> > m_tod_dotprod_ptsc1;
	tod_dotprod_scenario<10,4,medium_t<2,2,2> > m_tod_dotprod_ptsc2;
	tod_dotprod_scenario<5,4,large_t<2,2,2> > m_tod_dotprod_ptsc3;
public:
	//!	Creates the suite
	libtensor_pt_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_PT_SUITE_H

