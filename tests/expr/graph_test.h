#ifndef LIBTENSOR_GRAPH_TEST_H
#define LIBTENSOR_GRAPH_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::graph class

    \ingroup libtensor_tests_expr
**/
class graph_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_GRAPH_TEST_H

