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
    void test_construct_all_ignored() throw(libtest::test_exception);
    void test_construct_duplicate_subspaces_looped() throw(libtest::test_exception);

    void test_run_block_kernel_permute_2d() throw(libtest::test_exception);
    void test_run_block_kernel_permute_2d_sparse() throw(libtest::test_exception);
    void test_run_block_kernel_permute_3d_120() throw(libtest::test_exception);
    void test_run_block_kernel_permute_3d_120_sparse() throw(libtest::test_exception);

    void test_run_block_contract2_kernel_2d_2d() throw(libtest::test_exception);
    void test_run_block_contract2_kernel_3d_2d() throw(libtest::test_exception);

    void test_run_direct_3d_3d() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSE_LOOP_LIST_TEST_H_ */
