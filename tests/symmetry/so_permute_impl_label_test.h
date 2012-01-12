#ifndef LIBTENSOR_SO_PERMUTE_IMPL_LABEL_TEST_H
#define LIBTENSOR_SO_PERMUTE_IMPL_LABEL_TEST_H

#include "se_label_test_base.h"

namespace libtensor {


/**	\brief Tests the class libtensor::
        symmetry_operation_impl< so_permute<N, T>, se_label<N, T> >

	\ingroup libtensor_tests_sym
 **/
class so_permute_impl_label_test : public se_label_test_base {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1(const std::string &table_id) throw(libtest::test_exception);

	using se_label_test_base::check_allowed;
	using se_label_test_base::setup_pg_table;
};


} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_IMPL_LABEL_TEST_H

