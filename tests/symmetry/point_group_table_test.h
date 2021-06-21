#ifndef LIBTENSOR_POINT_GROUP_TABLE_TEST_H
#define LIBTENSOR_POINT_GROUP_TABLE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::point_group_table class

    \ingroup libtensor_tests_sym
 **/
class point_group_table_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
};


} // namespace libtensor

#endif // LIBTENSOR_POINT_GROUP_TABLE_TEST_H
