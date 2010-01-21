#ifndef LIBTENSOR_BTOD_SET_DIAG_TEST_H
#define LIBTENSOR_BTOD_SET_DIAG_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/block_index_space.h>

namespace libtensor {

/**	\brief Tests the libtensor::btod_set_diag class

	\ingroup libtensor_tests
**/
class btod_set_diag_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	template<size_t N>
	void run_test(const block_index_space<N> &bis, double d)
		throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_TEST_H
