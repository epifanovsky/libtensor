#ifndef LIBTENSOR_BTOD_IMPORT_RAW_TEST_H
#define LIBTENSOR_BTOD_IMPORT_RAW_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_index_space.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_import_raw class

	\ingroup libtensor_tests_btod
 **/
class btod_import_raw_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<size_t N>
	void test_1(const block_index_space<N> &bis)
	throw(libtest::test_exception);

	template<size_t N>
	void test_2(const block_index_space<N> &bis)
		throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_IMPORT_RAW_TEST_H
