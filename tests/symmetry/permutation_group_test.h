#ifndef LIBTENSOR_PERMUTATION_GROUP_TEST_H
#define LIBTENSOR_PERMUTATION_GROUP_TEST_H

#include <list>
#include <libtest/unit_test.h>
#include <libtensor/core/permutation.h>
#include <libtensor/symmetry/permutation_group.h>

namespace libtensor {


/** \brief Tests the libtensor::permutation_group class

    \ingroup libtensor_tests_sym
 **/
class permutation_group_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2a();
    void test_2b();
    void test_3();
    void test_4();
    void test_5a();
    void test_5b();
    void test_6a();
    void test_6b();
    void test_7();
    void test_8();

    void test_project_down_1();
    void test_project_down_2();
    void test_project_down_3();
    void test_project_down_4();
    void test_project_down_8a();
    void test_project_down_8b();

    void test_stabilize_1();
    void test_stabilize_2();
    void test_stabilize_3();
    void test_stabilize_4();
    void test_stabilize_5();
    void test_stabilize_6();
    void test_stabilize_7();

    void test_stabilize2_1();
    void test_stabilize2_2();
    void test_stabilize4_1();

    void test_permute_1();
    void test_permute_2();
    void test_permute_3();

private:
    typedef scalar_transf<double> transf_t;

    template<size_t N, typename T>
    void verify_group(const char *testname,
            const std::list< std::pair<permutation<N>, scalar_transf<T> > > &lst)
   ;

    template<size_t N, typename T>
    void verify_members(const char *testname,
            const permutation_group<N, T> &grp, const scalar_transf<T> &tr,
            const std::list< std::pair<permutation<N>, scalar_transf<T> > > &allowed)
   ;

    template<size_t N, typename T>
    void verify_genset(const char *testname,
            const permutation_group<N, T> &grp,
            const std::list< std::pair<permutation<N>, scalar_transf<T> > > &allowed)
   ;

    template<size_t N, typename T>
    void gen_group(
            const symmetry_element_set_adapter< N, T, se_perm<N, T> > &set,
            const scalar_transf<T> &tr, const permutation<N> &perm0,
            std::list< std::pair<permutation<N>, scalar_transf<T> > > &lst);
};


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_TEST_H
