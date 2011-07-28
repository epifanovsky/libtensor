#ifndef LIBTENSOR_SO_REDUCE_IMPL_PART_TEST_H
#define LIBTENSOR_SO_REDUCE_IMPL_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the class
        libtensor::symmetry_operation_impl<
            libtensor::so_reduce<N, M, K, T>, libtensor::se_part<N, T> >

	\ingroup libtensor_tests_sym
 **/
class so_reduce_impl_part_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_empty_1() throw(libtest::test_exception);
	void test_empty_2() throw(libtest::test_exception);
	void test_nm1_1(bool sign) throw(libtest::test_exception);
    void test_nm1_2(bool s1, bool s2) throw(libtest::test_exception);
    void test_nm1_3(bool sign) throw(libtest::test_exception);
    void test_nm1_4(bool sign) throw(libtest::test_exception);
    void test_nm1_5(bool sign) throw(libtest::test_exception);
    void test_nm1_6(bool sign) throw(libtest::test_exception);
    void test_nmk_1(bool sign) throw(libtest::test_exception);
    void test_nmk_2(bool s1, bool s2) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_STABILIZE_IMPL_PART_TEST_H

