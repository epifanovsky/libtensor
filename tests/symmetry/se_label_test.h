#ifndef LIBTENSOR_SE_LABEL_TEST_H
#define LIBTENSOR_SE_LABEL_TEST_H

#include <vector>
#include <libtest/unit_test.h>
#include <libtensor/symmetry/se_label.h>


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
    void test_allowed_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_allowed_2(
            const std::string &table_id) throw(libtest::test_exception);
    void test_allowed_3(
            const std::string &table_id) throw(libtest::test_exception);
    void test_permute_1(
            const std::string &table_id) throw(libtest::test_exception);
    void test_permute_2(
            const std::string &table_id) throw(libtest::test_exception);

    std::string setup_s6_symmetry();

    template<size_t N>
    void check_allowed(const char *testname, const char *sename, 
            const se_label<N, double> &se, const std::vector<bool> &expected) 
        throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_H

