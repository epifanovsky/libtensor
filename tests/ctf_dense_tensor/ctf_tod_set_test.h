#ifndef LIBTENSOR_CTF_TOD_SET_TEST_H
#define LIBTENSOR_CTF_TOD_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_tod_set class

    \ingroup libtensor_ctf_dense_tensor_tests
**/
class ctf_tod_set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_TEST_H

