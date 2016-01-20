#ifndef LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_TEST_H
#define LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::gen_bto_symcontract2_sym class

    \ingroup libtensor_tests_btod
**/
class gen_bto_symcontract2_sym_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_TEST_H
