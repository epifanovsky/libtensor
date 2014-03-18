#ifndef LIBTENSOR_CTF_BTOD_MULT_TEST_H
#define LIBTENSOR_CTF_BTOD_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_btod_mult class

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class ctf_btod_mult_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_mult_1();
    void test_div_1();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_MULT_TEST_H

