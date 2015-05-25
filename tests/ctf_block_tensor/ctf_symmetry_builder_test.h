#ifndef LIBTENSOR_CTF_SYMMETRY_BUILDER_TEST_H
#define LIBTENSOR_CTF_SYMMETRY_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::ctf_symmetry_builder class

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class ctf_symmetry_builder_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2a();
    void test_2b();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_BUILDER_TEST_H

