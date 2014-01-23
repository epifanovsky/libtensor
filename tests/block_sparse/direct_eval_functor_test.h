/*
 * direct_eval_functor_test.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef DIRECT_EVAL_FUNCTOR_TEST_H_
#define DIRECT_EVAL_FUNCTOR_TEST_H_

#include <libtest/unit_test.h>

namespace libtensor
{

class direct_eval_functor_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_contract_then_subtract() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* DIRECT_EVAL_FUNCTOR_TEST_H_ */
