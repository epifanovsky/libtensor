#ifndef LIBTENSOR_CTF_SYMMETRY_TEST_H
#define LIBTENSOR_CTF_SYMMETRY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_symmetry class

    \ingroup libtensor_ctf_dense_tensor_tests
 **/
class ctf_symmetry_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_convfac_1();
    void test_convfac_2();
    void test_convfac_3();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_TEST_H

