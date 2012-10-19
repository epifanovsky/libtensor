#ifndef LIBTENSOR_PERFORMANCE_TEST_H
#define LIBTENSOR_PERFORMANCE_TEST_H

#include <libtest/libtest.h>

namespace libtensor {


/** \brief Performance tests base class
    \tparam Repeats Number of times a performance test is repeated to get
        a meaningful result

    \ingroup libtensor_performance_tests
 **/
template<size_t Repeats>
class performance_test : public libtest::unit_test {
protected:
    virtual void do_calculate() = 0;

public:
    virtual void perform() throw(libtest::test_exception)
    {
        for ( size_t i=0; i<Repeats; i++ ) do_calculate();
    }
};


}

#endif // LIBTENSOR_PERFORMANCE_TEST_H
