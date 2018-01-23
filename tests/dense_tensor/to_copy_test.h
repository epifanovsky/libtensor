#ifndef LIBTENSOR_TO_COPY_TEST_H
#define LIBTENSOR_TO_COPY_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {


/** \brief Tests the libtensor::to_copy class

    \ingroup libtensor_tests_tod
 **/
class to_copy_test: public libtest::unit_test {
public:
    virtual void perform() throw (libtest::test_exception);

private:
    /** \brief Runs tests for a specific type T (double or float)
     **/
    template<typename T>
     void perform_run() throw (libtest::test_exception);

    /** \brief Tests plain copying of a tensor
     **/
    template<size_t N, typename T>
    void test_plain(const dimensions<N> &dims) throw (libtest::test_exception);

    /** \brief Tests plain copying of a tensor (additive version)
     **/
    template<size_t N, typename T>
    void test_plain_additive(const dimensions<N> &dims, T d)
        throw (libtest::test_exception);

    /** \brief Tests scaled copying of a tensor
     **/
    template<size_t N, typename T>
    void test_scaled(const dimensions<N> &dims, T c)
        throw (libtest::test_exception);

    /** \brief Tests scaled copying of a tensor (additive version)
     **/
    template<size_t N, typename T>
    void test_scaled_additive(const dimensions<N> &dims, T c, T d)
        throw (libtest::test_exception);

    /** \brief Tests permuted copying of a tensor
     **/
    template<size_t N, typename T>
    void test_perm(const dimensions<N> &dims, const permutation<N> &perm)
        throw (libtest::test_exception);

    /** \brief Tests permuted copying of a tensor (additive version)
     **/
    template<size_t N, typename T>
    void test_perm_additive(const dimensions<N> &dims,
        const permutation<N> &perm, T d) throw (libtest::test_exception);

    /** \brief Tests permuted and scaled copying of a tensor
     **/
    template<size_t N, typename T>
    void test_perm_scaled(const dimensions<N> &dims,
        const permutation<N> &perm, T c) throw (libtest::test_exception);

    /** \brief Tests permuted and scaled copying of a tensor
            (additive version)
     **/
    template<size_t N, typename T>
    void test_perm_scaled_additive(const dimensions<N> &dims,
        const permutation<N> &perm, T c, T d)
        throw (libtest::test_exception);

    /** \brief Tests if an exception is throws when the tensors have
            different dimensions
     **/
    template<typename T>
    void test_exc() throw (libtest::test_exception);

};


}; // namespace libtensor

#endif // LIBTENSOR_TO_COPY_TEST_H

