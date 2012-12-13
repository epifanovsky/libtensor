#ifndef LIBTENSOR_TOD_DOTPROD_SCENARIO_H
#define LIBTENSOR_TOD_DOTPROD_SCENARIO_H

#include <libtest/libtest.h>
#include "tod_dotprod_performance.h"
#include "performance_test_scenario_i.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \brief Performance test scenario for the libtensor::tod_dotprod class

    \param N dimensions of the tensors to be multiplied
    \param X size of the tensors

    All tests determine the size of the tensors by function dimA() of the X
    object.

    \ingroup libtensor_performance_tests
**/
template<size_t Repeats, size_t N, typename X>
class tod_dotprod_scenario : public performance_test_scenario_i {

private:
    unit_test_factory<tod_dotprod_ref<Repeats,X> > m_ref;
    unit_test_factory<tod_dotprod_p1<Repeats,N,X> > m_pt1;
    unit_test_factory<tod_dotprod_p2<Repeats,N,X> > m_pt2;
    unit_test_factory<tod_dotprod_p3<Repeats,N,X> > m_pt3;

public:
    tod_dotprod_scenario();

    virtual ~tod_dotprod_scenario() {}
};


template<size_t Repeats, size_t N, typename X>
tod_dotprod_scenario<Repeats,N,X>::tod_dotprod_scenario() {

    add_test("reference","<A,B>",m_ref);
    add_test("test 1","<A,B>",m_pt1);
    add_test("test 2","<A,P_I B>",m_pt2);
    add_test("test 3","<A,B'>",m_pt3);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_SCENARIO_H

