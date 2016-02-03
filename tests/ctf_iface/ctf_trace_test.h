#ifndef LIBTENSOR_CTF_TRACE_TEST_H
#define LIBTENSOR_CTF_TRACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::trace function

    \ingroup libtensor_tests_ctf_iface
 **/
class ctf_trace_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_t_1();
    void test_t_2();
    void test_t_3();
    void test_e_1();
    void test_e_2();
    void test_e_3();

};

} // namespace libtensor

#endif // LIBTENSOR_CTF_TRACE_TEST_H
