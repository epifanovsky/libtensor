#ifndef LIBTENSOR_SO_MERGE_TEST_H
#define LIBTENSOR_SO_MERGE_TEST_H

#include "se_label_test_base.h"

namespace libtensor {

/** \brief Tests the libtensor::so_merge class

    \ingroup libtensor_tests_sym
**/
class so_merge_test : public se_label_test_base {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();

};

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_TEST_H
