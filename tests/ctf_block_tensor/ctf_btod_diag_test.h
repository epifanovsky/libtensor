#ifndef LIBTENSOR_CTF_BTOD_DIAG_TEST_H
#define LIBTENSOR_CTF_BTOD_DIAG_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_btod_diag class

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class ctf_btod_diag_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a();
    void test_1b();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_DIAG_TEST_H

