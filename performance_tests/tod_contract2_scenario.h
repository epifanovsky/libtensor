#ifndef TOD_CONTRACT_SCENARIO_H
#define TOD_CONTRACT_SCENARIO_H

#include <libtest/libtest.h>
#include "tod_contract2_performance.h"
#include "performance_test_scenario_i.h"

using libtest::unit_test_factory;

namespace libtensor {

/**	\brief Performance test scenario for the libtensor::tod_add class

 	\param N dimensions of the tensors to be added
 	\param X size of the tensors

	All tests determine the size of the tensors A, B, and C by functions dimA(),
	dimB(),	and dimC() of the X object, respectively.

 	\ingroup libtensor_performance_tests
**/
template<size_t Repeats,size_t N, size_t M, size_t K, typename X>
class tod_contract2_scenario
	: public performance_test_scenario_i
{
	unit_test_factory<tod_contract2_ref<Repeats,N,M,K,X> > m_ref;
	unit_test_factory<tod_contract2_p1<Repeats,N,M,K,X> > m_pt1;
	unit_test_factory<tod_contract2_p2<Repeats,N,M,K,X> > m_pt2;
	unit_test_factory<tod_contract2_p3<Repeats,N,M,K,X> > m_pt3;
	unit_test_factory<tod_contract2_p4<Repeats,N,M,K,X> > m_pt4;

public:
	tod_contract2_scenario();
	virtual ~tod_contract2_scenario() {}
};


template<size_t Repeats,size_t N, size_t M, size_t K, typename X>
tod_contract2_scenario<Repeats,N,M,K,X>::tod_contract2_scenario()
{
	add_test("reference",
			"A_{i_1,i_2} += 0.5 \\sum_{i_3} B_{i_1,i_3} C_{i_2,i_3}",m_ref);
	add_test("test 1",
			"A_{i_1,i_2} += 0.5 \\sum_{i_3} B_{i_1,i_3} C_{i_2,i_3}",m_pt1);
	add_test("test 2",
			"A_{i_1,i_2} += 0.5 \\sum_{i_3} B_{i_1,i_3} C_{i_3,i_2}",m_pt2);
	add_test("test 3",
			"A_{i} += 0.5 P_I \\sum_{i_3} B_{i_1,i_3} C_{i_3,i_2}",m_pt3);
	add_test("test 4",
			"A_{i_1,i_2} += 0.5 \\sum_{i_3} B_{i_1,i_3} P_3 C_{i_2,i_3}",m_pt4);
}


} // namespace libtensor

#endif // TOD_ADD_SCENARIO_H

