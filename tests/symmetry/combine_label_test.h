#ifndef LIBTENSOR_COMBINE_LABEL_TEST_H
#define LIBTENSOR_COMBINE_LABEL_TEST_H

#include "se_label_test_base.h"


namespace libtensor {

/** \brief Tests the libtensor::combine_label class

    \ingroup libtensor_tests_sym
 **/
class combine_label_test : public se_label_test_base {
public:
    virtual void perform();

private:
    void test_1(
            const std::string &table_id);
    void test_2(
            const std::string &table_id);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;
};

} // namespace libtensor

#endif // LIBTENSOR_COMBINE_LABEL_TEST_H

