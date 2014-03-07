/*
 * sparse_loop_list_test.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_LOOP_GROUPER_TEST_H_
#define SPARSE_LOOP_GROUPER_TEST_H_

#include <libtest/unit_test.h>

namespace libtensor
{

class sparse_loop_grouper_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_get_n_groups() throw(libtest::test_exception);
    void test_get_bispaces_and_index_groups() throw(libtest::test_exception);
    void test_get_offsets_and_sizes() throw(libtest::test_exception);
    void test_get_bispaces_and_subspaces() throw(libtest::test_exception);
    void test_get_block_dims() throw(libtest::test_exception);
    void test_get_loops_for_groups() throw(libtest::test_exception);
    void test_get_offsets_and_sizes_C_direct() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSE_LOOP_GROUPER_TEST_H_ */
