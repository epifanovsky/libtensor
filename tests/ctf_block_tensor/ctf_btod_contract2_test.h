#ifndef LIBTENSOR_CTF_BTOD_CONTRACT2_TEST_H
#define LIBTENSOR_CTF_BTOD_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_btod_contract2 class

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class ctf_btod_contract2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2(double d);
    void test_3(double d);
    void test_4(double d);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_CONTRACT2_TEST_H

