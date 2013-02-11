#ifndef LIBTENSOR_CTF_TOD_COPY_TEST_H
#define LIBTENSOR_CTF_TOD_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_tod_copy class

    \ingroup libtensor_ctf_dense_tensor_tests
**/
class ctf_tod_copy_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a();
    void test_1b();
    void test_2a();
    void test_2b();
    void test_3a();
    void test_3b();
    void test_4a();
    void test_4b();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_COPY_TEST_H

