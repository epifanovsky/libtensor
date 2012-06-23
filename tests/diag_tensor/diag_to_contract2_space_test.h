#ifndef LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_TEST_H
#define LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_to_contract2_space class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_to_contract2_space_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a() throw(libtest::test_exception);
    void test_1b() throw(libtest::test_exception);
    void test_1c() throw(libtest::test_exception);
    void test_2a() throw(libtest::test_exception);
    void test_3a() throw(libtest::test_exception);
    void test_4a() throw(libtest::test_exception);
    void test_4b() throw(libtest::test_exception);
    void test_4c() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_CONTRACT2_SPACE_TEST_H

