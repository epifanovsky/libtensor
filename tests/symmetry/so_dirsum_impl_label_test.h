#ifndef LIBTENSOR_SO_DIRSUM_IMPL_LABEL_TEST_H
#define LIBTENSOR_SO_DIRSUM_IMPL_LABEL_TEST_H

#include "se_label_test_base.h"

namespace libtensor {


/**	\brief Tests the class libtensor::
        symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> >

	\ingroup libtensor_tests_sym
 **/
class so_dirsum_impl_label_test : public se_label_test_base {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_empty_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_empty_2(const std::string &table_id,
            bool perm) throw(libtest::test_exception);
    void test_empty_3(const std::string &table_id,
            bool perm) throw(libtest::test_exception);
    void test_nn_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_nn_2(
            const std::string &table_id) throw(libtest::test_exception);
    void test_nn_3(
            const std::string &table_id) throw(libtest::test_exception);

    using se_label_test_base::setup_pg_table;
    using se_label_test_base::check_allowed;
};


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_IMPL_LABEL_TEST_H

