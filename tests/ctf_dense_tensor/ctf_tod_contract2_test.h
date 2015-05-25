#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_TEST_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_tod_contract2 class

    \ingroup libtensor_ctf_dense_tensor_tests
**/
class ctf_tod_contract2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1(double d);
    void test_2(double d);
    void test_3a();
    void test_3b();
    void test_4(double d);
    void test_5(double d);
    void test_6(double d);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_TEST_H

