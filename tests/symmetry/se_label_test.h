#ifndef LIBTENSOR_SE_LABEL_TEST_H
#define LIBTENSOR_SE_LABEL_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::se_label class

	\ingroup libtensor_tests_sym
 **/
class se_label_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_basic_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_1(
            const std::string &table_id) throw(libtest::test_exception);

    const std::string &setup_s6_symmetry() const;

    template<size_t N>
    void check_allowed(const char *testname, const se_label<N, double> &se,
            const std::vector<bool> &expected) throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_H

