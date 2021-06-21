#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_PART_TEST_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the class
        libtensor::symmetry_operation_impl<
            libtensor::so_symmetrize<N, T>, libtensor::se_part<N, T> >

    \ingroup libtensor_tests_sym
 **/
class so_symmetrize_se_part_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_empty_1();
    void test_empty_2();
    void test_sym2_1(bool sign);
    void test_sym2_2();

};


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PART_TEST_H

