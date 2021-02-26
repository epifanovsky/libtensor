#ifndef LIBTENSOR_TO_ADD_TEST_H
#define LIBTENSOR_TO_ADD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::to_add class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_add_test_x : public libtest::unit_test {
    static const T k_thresh; //!< Threshold multiplier
public:
    virtual void perform() throw(libtest::test_exception);
private:
    /** \brief Tests if exceptions are thrown when the tensors have
            different dimensions
    **/
    void test_exc() throw(libtest::test_exception);

    /** \brief Tests addition of a tensor to itself

        \f[ T_{pqrs} = 2.0 A_{pqrs} + 0.5 A_{pqrs}  \f]
    **/
    void test_add_to_self_pqrs(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of two tensors (no permutation)

        \f[ T_{pqrs} = T_{pqrs} + 0.1 A_{pqrs}  \f]
    **/
    void test_add_two_pqrs_pqrs(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of two tensors (permutation type 1)

        \f[ T_{pqrs} = T_{pqrs} + 0.1 A_{qprs}  \f]
    **/
    void test_add_two_pqrs_qprs(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of two tensors (permutation type 2)

        \f[ T_{pqrs} = T_{pqrs} + 0.1 A_{prsq}  \f]
    **/
    void test_add_two_pqrs_prsq(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of two tensors (permutation type 3)

        \f[ T_{pqrs} = T_{pqrs} + 0.1 A_{qpsr}  \f]
    **/
    void test_add_two_pqrs_qpsr(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of three tensors

        \f[ T_{pqrs} = T_{pqrs} + 0.5 \left( A_{pqrs} - 4.0 B_{qprs} \right) \f]
    **/
    void test_add_mult(size_t, size_t, size_t, size_t)
        throw(libtest::test_exception);

    /** \brief Tests addition of three tensors (in two dimensions)

        \f[ T_{pq} = T_{pq} + 0.5 \left( 2.0 A_{pq} - B_{qp} \right) \f]
    **/
    void test_add_two_pq_qp( size_t, size_t )
        throw(libtest::test_exception);

    void test_add_two_ijkl_kjli(size_t ni, size_t nj, size_t nk, size_t nl,
        T c1, T c2) throw(libtest::test_exception);

};

class to_add_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_ADD_TEST_H

