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
    void test_get_loops_for_tree() throw(libtest::test_exception);
    void test_get_trees_for_loop() throw(libtest::test_exception);
    void test_get_bispaces_and_index_groups_for_tree() throw(libtest::test_exception);
    void test_get_sub_key_offsets_for_tree() throw(libtest::test_exception);
    void test_fuse() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSITY_FUSER_TEST_H_ */
