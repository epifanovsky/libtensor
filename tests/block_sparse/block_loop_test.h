#ifndef BLOCK_LOOP_TEST_H
#define BLOCK_LOOP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class block_loop_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private: 

    void test_set_subspace_looped_invalid_bispace_idx() throw(libtest::test_exception);
    void test_set_subspace_looped_invalid_subspace_idx() throw(libtest::test_exception);
    void test_set_subspace_looped_not_matching_subspaces() throw(libtest::test_exception);

    void test_get_subspace_looped_invalid_bispace_idx() throw(libtest::test_exception);
    void test_get_subspace_looped() throw(libtest::test_exception);

    void test_is_bispace_ignored_invalid_bispace_idx() throw(libtest::test_exception);
    void test_is_bispace_ignored() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* BLOCK_LOOP_TEST_H */
