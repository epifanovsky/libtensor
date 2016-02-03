#ifndef LIBTENSOR_CTF_TOD_MULT1_TEST_H
#define LIBTENSOR_CTF_TOD_MULT1_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_tod_mult1 class

    \ingroup libtensor_ctf_dense_tensor_tests
 **/
class ctf_tod_mult1_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_mult_1a();
    void test_mult_1b();
    void test_div_1a();
    void test_div_1b();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_MULT1_TEST_H

