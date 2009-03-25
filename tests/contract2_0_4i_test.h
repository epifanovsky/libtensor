/*
 * contract2_0_4i_test.h
 *
 *  Created on: Mar 23, 2009
 *      Author: kirhist
 */

#include <libtest.h>
#include "contract2_0_4i.h"

#ifndef LIBTENSOR_CONTRACT2_0_4I_TEST_H
#define LIBTENSOR_CONTRACT2_0_4I_TEST_H

namespace libtensor {

/**	\brief Tests the libtensor::contract2_0_4i class

	\ingroup libtensor_tests
**/
class contract2_0_4i_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	void test_ijkl_ijkl(size_t ni, size_t nj, size_t nk, size_t nl)
									throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* LIBTENSOR_CONTRACT2_0_4I_TEST_H */
