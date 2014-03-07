/*
 * sparse_loop_list_test.h
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#ifndef SPARSE_LOOP_ITERATOR_TEST_H_
#define SPARSE_LOOP_ITERATOR_TEST_H_

#include <libtest/unit_test.h>

namespace libtensor
{

class sparse_loop_iterator_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_set_offsets_and_dims_dense() throw(libtest::test_exception);
    void test_increment_dense() throw(libtest::test_exception);

    void test_set_offsets_and_dims_sparse() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* SPARSE_LOOP_ITERATOR_TEST_H_ */
