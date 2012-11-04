#ifndef LIBTENSOR_BLOCK_MAP_TEST_H
#define LIBTENSOR_BLOCK_MAP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::block_map class

    \ingroup libtensor_tests_core
 **/
class block_map_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_create();
    void test_immutable();
    void test_get_all_1();

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_TEST_H
