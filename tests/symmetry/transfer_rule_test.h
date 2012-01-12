#ifndef LIBTENSOR_TRANSFER_RULE_TEST_H
#define LIBTENSOR_TRANSFER_RULE_TEST_H

#include <libtensor/symmetry/label/evaluation_rule.h>
#include "se_label_test_base.h"

namespace libtensor {

/**	\brief Tests the libtensor::transfer_rule class

	\ingroup libtensor_tests_sym
 **/
class transfer_rule_test : public se_label_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_basic_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_basic_2(
            const std::string &table_id) throw(libtest::test_exception);
    void test_basic_3(
            const std::string &table_id) throw(libtest::test_exception);
    void test_merge_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_merge_2(
            const std::string &table_id) throw(libtest::test_exception);
    void test_merge_3(
            const std::string &table_id) throw(libtest::test_exception);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;


};

} // namespace libtensor

#endif // LIBTENSOR_TRANSFER_RULE_TEST_H

