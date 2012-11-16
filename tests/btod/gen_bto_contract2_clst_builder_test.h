#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_TEST_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::gen_bto_contract2_clst_builder class

    \ingroup libtensor_tests_btod
 **/
class gen_bto_contract2_clst_builder_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_TEST_H
