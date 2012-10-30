#ifndef LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_TEST_H
#define LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::gen_bto_unfold_symmetry class

    \ingroup libtensor_tests_btod
 **/
class gen_bto_unfold_symmetry_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_TEST_H
