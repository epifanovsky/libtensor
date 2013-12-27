/*
 * sparse_loop_list_test.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSITY_FUSER_TEST_H_
#define SPARSITY_FUSER_TEST_H_

#include <libtest/unit_test.h>

namespace libtensor
{

class sparsity_fuser_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_get_loops_accessing_tree() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSITY_FUSER_TEST_H_ */
