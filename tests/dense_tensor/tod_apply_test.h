#ifndef LIBTENSOR_TOD_APPLY_TEST_H
#define LIBTENSOR_TOD_APPLY_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {

/** \brief Tests the libtensor::tod_apply class

    \ingroup libtensor_tests_tod
**/
class tod_apply_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    /** \brief Tests plain applying a functor to a %tensor
     **/
    template<size_t N, typename Functor>
    void test_plain(Functor &fn, const dimensions<N> &dims)
        throw(libtest::test_exception);

    /** \brief Tests plain applying a functor to a %tensor
            (additive version)
     **/
    template<size_t N, typename Functor>
    void test_plain_additive(Functor &fn, const dimensions<N> &dims, double d)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a scaled %tensor
     **/
    template<size_t N, typename Functor>
    void test_scaled(Functor &fn, const dimensions<N> &dims, double c)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a scaled %tensor
            (additive version)
     **/
    template<size_t N, typename Functor>
    void test_scaled_additive(Functor &fn, const dimensions<N> &dims,
        double c, double d)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a permuted %tensor
     **/
    template<size_t N, typename Functor>
    void test_perm(Functor &fn, const dimensions<N> &dims,
        const permutation<N> &perm)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a permuted %tensor
            (additive version)
     **/
    template<size_t N, typename Functor>
    void test_perm_additive(Functor &fn, const dimensions<N> &dims,
        const permutation<N> &perm, double d)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a permuted and scaled %tensor
     **/
    template<size_t N, typename Functor>
    void test_perm_scaled(Functor &fn, const dimensions<N> &dims,
        const permutation<N> &perm, double c)
        throw(libtest::test_exception);

    /** \brief Tests applying a functor to a permuted and scaled %tensor
            (additive version)
     **/
    template<size_t N, typename Functor>
    void test_perm_scaled_additive(Functor &fn, const dimensions<N> &dims,
        const permutation<N> &perm, double c, double d)
        throw(libtest::test_exception);

    /** \brief Tests if an exception is thrown when the tensors have
            different dimensions
     **/
    void test_exc() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_APPLY_TEST_H

