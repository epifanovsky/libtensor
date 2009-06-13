/*
 * tod_solve_test.h
 *
 *  Created on: Apr 17, 2009
 *      Author: kirhist
 */

#ifndef TOD_SOLVE_TEST_H_
#define TOD_SOLVE_TEST_H_

#include <libtest.h>
//#include "tod_solve.h"

namespace libtensor {

	/**	\brief Tests the libtensor::tod_add class

		\ingroup libtensor_tests
	**/
	class tod_solve_test : public libtest::unit_test {
	public:
		virtual void perform() throw(libtest::test_exception);
	};

} // namespace libtensor

#endif /* TOD_SOLVE_TEST_H_ */
