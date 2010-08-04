#ifndef LIBTENSOR_PERMUTATION_GROUP_TEST_H
#define LIBTENSOR_PERMUTATION_GROUP_TEST_H

#include <list>
#include <libtest/unit_test.h>
#include <libtensor/core/permutation.h>
#include <libtensor/symmetry/permutation_group.h>

namespace libtensor {


/**	\brief Tests the libtensor::permutation_group class

	\ingroup libtensor_tests
 **/
class permutation_group_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_1() throw(libtest::test_exception);
	void test_2a() throw(libtest::test_exception);
	void test_2b() throw(libtest::test_exception);
	void test_3() throw(libtest::test_exception);
	void test_4() throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);
	void test_6a() throw(libtest::test_exception);
	void test_6b() throw(libtest::test_exception);
	void test_7() throw(libtest::test_exception);
	void test_8() throw(libtest::test_exception);

	void test_project_down_1() throw(libtest::test_exception);
	void test_project_down_2() throw(libtest::test_exception);
	void test_project_down_3() throw(libtest::test_exception);
	void test_project_down_4() throw(libtest::test_exception);
	void test_project_down_8a() throw(libtest::test_exception);
	void test_project_down_8b() throw(libtest::test_exception);

	void test_stabilize_1() throw(libtest::test_exception);
	void test_stabilize_2() throw(libtest::test_exception);
	void test_stabilize_3() throw(libtest::test_exception);
	void test_stabilize_4() throw(libtest::test_exception);
	void test_stabilize_5() throw(libtest::test_exception);
	void test_stabilize_6() throw(libtest::test_exception);
	void test_stabilize_7() throw(libtest::test_exception);

	void test_stabilize2_1() throw(libtest::test_exception);
	void test_stabilize2_2() throw(libtest::test_exception);

	void test_permute_1() throw(libtest::test_exception);
	void test_permute_2() throw(libtest::test_exception);
	void test_permute_3() throw(libtest::test_exception);

	template<size_t N>
	void verify_group(const char *testname,
		const std::list< permutation<N> > &lst)
		throw(libtest::test_exception);

	template<size_t N, typename T>
	void verify_members(const char *testname,
		const permutation_group<N, T> &grp,
		const std::list< permutation<N> > &allowed_symm,
		const std::list< permutation<N> > &allowed_asymm)
		throw(libtest::test_exception);

	template<size_t N, typename T>
	void verify_genset(const char *testname,
		const permutation_group<N, T> &grp,
		const std::list< permutation<N> > &allowed_symm,
		const std::list< permutation<N> > &allowed_asymm)
		throw(libtest::test_exception);

	template<size_t N>
	void all_permutations(std::list< permutation<N> > &lst);
	void all_permutations(std::list< permutation<1> > &lst);
	void all_permutations(std::list< permutation<0> > &lst);

	template<size_t N, typename T>
	void gen_group(
		const symmetry_element_set_adapter< N, T, se_perm<N, T> > &set,
		bool sign, const permutation<N> &perm0,
		std::list< permutation<N> > &lst);
};


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_TEST_H
