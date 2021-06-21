#ifndef LIBTENSOR_SO_MERGE_SE_PART_TEST_H
#define LIBTENSOR_SO_MERGE_SE_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::so_merge_se_part class

    \ingroup libtensor_tests_sym
 **/
class so_merge_se_part_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_empty_1();
    void test_empty_2();
    void test_nm1_1(bool sign);
    void test_nm1_2(bool sign);
    void test_2n2nn_1(bool s1, bool s2);
    void test_2n2nn_2(bool s1, bool s2);
    void test_2n2nn_3(bool sign);
    void test_nmk_1(bool sign);
    void test_nmk_2(bool s1, bool s2);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_PART_TEST_H

