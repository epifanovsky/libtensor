#ifndef LIBTENSOR_SE_LABEL_TEST_H
#define LIBTENSOR_SE_LABEL_TEST_H

#include "se_label_test_base.h"


namespace libtensor {

/** \brief Tests the libtensor::se_label class

    \ingroup libtensor_tests_sym
 **/
class se_label_test : public se_label_test_base {
public:
    virtual void perform();

private:
    void test_basic_1(
            const std::string &table_id);
    void test_allowed_1(
            const std::string &table_id);
    void test_allowed_2(
            const std::string &table_id);
    void test_allowed_3(
            const std::string &table_id);
    void test_permute_1(
            const std::string &table_id);
    void test_permute_2(
            const std::string &table_id);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;
};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_H

