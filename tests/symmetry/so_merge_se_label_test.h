#ifndef LIBTENSOR_SO_MERGE_SE_LABEL_TEST_H
#define LIBTENSOR_SO_MERGE_SE_LABEL_TEST_H

#include "se_label_test_base.h"

namespace libtensor {

/**	\brief Tests the libtensor::so_merge_se_label class

	\ingroup libtensor_tests_sym
 **/
class so_merge_se_label_test : public se_label_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_empty_1(
	        const std::string &table_id) throw(libtest::test_exception);
    void test_empty_2(
            const std::string &table_id) throw(libtest::test_exception);
	void test_nm1_1(
	        const std::string &table_id) throw(libtest::test_exception);
	void test_nm1_2(
	        const std::string &table_id) throw(libtest::test_exception);
	void test_2n2nn_1(
	        const std::string &table_id) throw(libtest::test_exception);
    void test_2n2nn_2(const std::string &table_id,
            bool product) throw(libtest::test_exception);
    void test_nmk_1(const std::string &table_id,
            bool product) throw(libtest::test_exception);
    void test_nmk_2(const std::string &table_id,
            bool product) throw(libtest::test_exception);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;
};


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_LABEL_TEST_H

