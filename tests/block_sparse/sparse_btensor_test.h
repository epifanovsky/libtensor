#ifndef SPARSE_BTENSOR_TEST_H
#define SPARSE_BTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::sparse_btensor class

    \ingroup libtensor_tests_sparse
**/
class sparse_btensor_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private:

    /** \brief Returns the correct bispace object that was used to construct it?
    **/
    void test_get_bispace() throw(libtest::test_exception);

    /** \brief Test that equality returns true when appropriate
    **/
    void test_equality_true() throw(libtest::test_exception);

    /** \brief Correct string representation of the tensor
    **/
    void test_str() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* SPARSE_BTENSOR_TEST_H */
