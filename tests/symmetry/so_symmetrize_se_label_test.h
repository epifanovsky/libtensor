#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_TEST_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_TEST_H

#include "se_label_test_base.h"

namespace libtensor {

/** \brief Tests the libtensor::so_symmetrize_se_label class

    \ingroup libtensor_tests_sym
 **/
class so_symmetrize_se_label_test : public se_label_test_base {
public:
    virtual void perform();

private:
    void test_empty(
            const std::string &table_id);
    void test_sym2_1(
            const std::string &table_id);
    void test_sym2_2(
            const std::string &table_id);
    void test_sym2_3(
            const std::string &table_id);
    void test_sym3_1(
            const std::string &table_id);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;
};


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_TEST_H

