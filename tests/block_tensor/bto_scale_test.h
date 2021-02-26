#ifndef LIBTENSOR_BTO_SCALE_TEST_H
#define LIBTENSOR_BTO_SCALE_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {

/** \brief Tests the libtensor::bto_scale class

    \ingroup libtensor_tests_btod
**/
template<typename T>
class bto_scale_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

private:
    template<size_t N>
    void test_generic(const char *testname,
        block_tensor_i<N, T> &bt, T c)
        throw(libtest::test_exception);

    void test_0() throw(libtest::test_exception);
    void test_i(size_t i) throw(libtest::test_exception);

    void test_1() throw(libtest::test_exception);
};

class bto_scale_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_SCALE_TEST_H
