#ifndef LIBTENSOR_CTF_BTENSOR_TEST_H
#define LIBTENSOR_CTF_BTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_btensor class

    \ingroup libtensor_ctf_iface_tests
 **/
class ctf_btensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTENSOR_TEST_H

