#ifndef LIBTENSOR_ADJACENCY_LIST_TEST_H
#define LIBTENSOR_ADJACENCY_LIST_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::adjacency_list<N> class

    \ingroup libtensor_tests_sym
 **/
class adjacency_list_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
};

} // namespace libtensor

#endif // LIBTENSOR_ADJACENCY_LIST_TEST_H

