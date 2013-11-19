/*
 * sparse_loop_list_test.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_LOOP_LIST_TEST_H_
#define SPARSE_LOOP_LIST_TEST_H_

#include <libtest/unit_test.h>

namespace libtensor
{

class sparse_loop_list_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_add_loop_invalid_loop_bispaces() throw(libtest::test_exception);
    void test_add_loop_all_ignored() throw(libtest::test_exception);
    void test_add_loop_duplicate_subspaces_looped() throw(libtest::test_exception);

    void test_get_loops_that_access_bispace_invalid_bispace() throw(libtest::test_exception);
    void test_get_loops_that_access_bispace_2d_matmul() throw(libtest::test_exception);

    void test_run_block_permute_kernel_2d() throw(libtest::test_exception);
    void test_run_block_permute_kernel_2d_sparse() throw(libtest::test_exception);
    void test_run_block_permute_kernel_3d_120() throw(libtest::test_exception);
    void test_run_block_permute_kernel_3d_120_sparse() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSE_LOOP_LIST_TEST_H_ */
