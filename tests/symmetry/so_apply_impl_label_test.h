#ifndef LIBTENSOR_SO_APPLY_IMPL_LABEL_TEST_H
#define LIBTENSOR_SO_APPLY_IMPL_LABEL_TEST_H

#include "se_label_test_base.h"

namespace libtensor {


/**	\brief Tests the libtensor::so_apply_impl_label class

	\ingroup libtensor_tests_sym
 **/
class so_apply_impl_label_test : public se_label_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(const std::string &table_id, bool keep_zero,
	        bool is_asym, bool sign) throw(libtest::test_exception);

	using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;

};


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_LABEL_TEST_H

