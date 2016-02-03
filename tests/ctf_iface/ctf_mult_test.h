#ifndef LIBTENSOR_CTF_MULT_TEST_H
#define LIBTENSOR_CTF_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::mult function

    \ingroup libtensor_tests_ctf_iface
 **/
class ctf_mult_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_tt_1a();
    void test_tt_1b();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_MULT_TEST_H
