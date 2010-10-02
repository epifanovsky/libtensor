#ifndef LIBTENSOR_PARTITION_SET_TEST_H
#define LIBTENSOR_PARTITION_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::partition_set class

	\ingroup libtensor_tests
 **/
class partition_set_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	// Creating partitions
	void test_1() throw(libtest::test_exception);
	void test_2(bool sign) throw(libtest::test_exception);
	void test_3(bool sign) throw(libtest::test_exception);
	void test_4(bool sign) throw(libtest::test_exception);
	void test_5() throw(libtest::test_exception);

	// Adding partitions
	void test_add_1(bool sign) throw(libtest::test_exception);
	void test_add_2(bool sign) throw(libtest::test_exception);
	void test_add_3a(bool sign) throw(libtest::test_exception);
	void test_add_3b(bool sign) throw(libtest::test_exception);
	void test_add_4(bool sign) throw(libtest::test_exception);
	void test_add_5() throw(libtest::test_exception);
	void test_add_6(bool sign) throw(libtest::test_exception);
	void test_add_7(bool sign) throw(libtest::test_exception);
	void test_add_8(bool sign) throw(libtest::test_exception);

	// Permuting partitions
	void test_permute_1() throw(libtest::test_exception);
	void test_permute_2(bool sign) throw(libtest::test_exception);
	void test_permute_3(bool sign) throw(libtest::test_exception);

	// Intersection of partitions
	void test_intersect_1() throw(libtest::test_exception);
	void test_intersect_2(bool first_empty) throw(libtest::test_exception);
	void test_intersect_3(bool sign) throw(libtest::test_exception);
	void test_intersect_4(bool sign) throw(libtest::test_exception);
	void test_intersect_5a(bool sign) throw(libtest::test_exception);
	void test_intersect_5b(bool sign) throw(libtest::test_exception);
	void test_intersect_6a(bool sign) throw(libtest::test_exception);
	void test_intersect_6b(bool sign) throw(libtest::test_exception);
	void test_intersect_7a(bool sign) throw(libtest::test_exception);
	void test_intersect_7b(bool sign) throw(libtest::test_exception);
	void test_intersect_7c(bool sign) throw(libtest::test_exception);

	// Stabilizing dimensions
	void test_stabilize_1() throw(libtest::test_exception);
	void test_stabilize_2(bool sign) throw(libtest::test_exception);
	void test_stabilize_3(bool sign) throw(libtest::test_exception);
	void test_stabilize_4(bool sign) throw(libtest::test_exception);

	// Merging dimensions
	void test_merge_1() throw(libtest::test_exception);
	void test_merge_2(bool sign) throw(libtest::test_exception);
	void test_merge_3(bool sign) throw(libtest::test_exception);
	void test_merge_4(bool sign) throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_PARTITION_SET_TEST_H

