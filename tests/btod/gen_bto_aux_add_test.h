#ifndef LIBTENSOR_GEN_BTO_AUX_ADD_TEST_H
#define LIBTENSOR_GEN_BTO_AUX_ADD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::gen_bto_aux_add class

    \ingroup libtensor_tests_btod
 **/
class gen_bto_aux_add_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1a();
    void test_1b();
    void test_2();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_ADD_TEST_H
