#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_PRODUCT_BUILDER_TEST_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_PRODUCT_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::block_index_space_product_builder class

    \ingroup libtensor_tests_core
 **/
class block_index_space_product_builder_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_0a() throw(libtest::test_exception);
    void test_0b() throw(libtest::test_exception);
    void test_1a() throw(libtest::test_exception);
    void test_1b() throw(libtest::test_exception);
    void test_2a() throw(libtest::test_exception);
    void test_2b() throw(libtest::test_exception);
    void test_3a() throw(libtest::test_exception);
    void test_3b() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_PRODUCT_BUILDER_TEST_H
