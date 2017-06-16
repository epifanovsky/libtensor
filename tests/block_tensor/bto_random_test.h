#ifndef LIBTENSOR_BTO_RANDOM_TEST_H
#define LIBTENSOR_BTO_RANDOM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::bto_random class

    \ingroup libtensor_tests_btod
**/
template<typename T>
class bto_random_test_x : public libtest::unit_test {
public:
    static const T k_thresh;
    virtual void perform() throw(libtest::test_exception);

};

class bto_random_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTO_RANDOM_TEST_H
