#ifndef LIBTENSOR_GEN_BTO_AUX_COPY_TEST_H
#define LIBTENSOR_GEN_BTO_AUX_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::gen_bto_aux_copy class

    \ingroup libtensor_tests_btod
 **/
class gen_bto_aux_copy_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1a();
    void test_1b();
    void test_1c();
    void test_2();
    void test_exc_1();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_COPY_TEST_H
