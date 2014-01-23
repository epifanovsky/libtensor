/* * direct_eval_functor_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */
#include "direct_eval_functor_test.h"

using namespace std;

namespace libtensor {

void direct_eval_functor_test::perform() throw(libtest::test_exception) {
    test_contract_then_subtract();
}

void direct_eval_functor_test::test_contract_then_subtract() throw(libtest::test_exception)
{
}

} // namespace libtensor
