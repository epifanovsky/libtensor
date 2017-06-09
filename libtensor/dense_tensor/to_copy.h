#ifndef LIBTENSOR_TO_COPY_H
#define LIBTENSOR_TO_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Copies the contents of a tensor, permutes and scales the entries if
        necessary
    \tparam N Tensor order.

    This operation makes a transformed copy of a %tensor.
    The result can replace or be added to the output %tensor.

    <b>Examples</b>

    Plain copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    to_copy<2> cp(t1);
    cp.perform(t2); // Copies the elements of t1 to t2
    \endcode

    Scaled copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    to_copy<2> cp(t1, 0.5);
    cp.perform(t2); // Copies the elements of t1 multiplied by 0.5 to t2
    \endcode

    Permuted copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    to_copy<2> cp(t1, perm);
    cp.perform(t2); // Copies transposed t1 to t2
    \endcode

    Permuted and scaled copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    to_copy<2> cp(t1, perm, 0.5);
    cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
    \endcode
    or
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tensor_transf<2, double> tr(perm, 0.5);
    to_copy<2> cp(t1, tr);
    cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
    \endcode

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_copy : public timings< to_copy<N> >, public noncopyable {

public:
    static const char *k_clazz; //!< Class name

    typedef tensor_transf<N, T> tensor_transf_t;

private:
    dense_tensor_rd_i<N, T> &m_ta; //!< Source tensor
    permutation<N> m_perm; //!< Permutation of indexes
    T m_c; //!< Scaling coefficient
    dimensions<N> m_dimsb; //!< Dimensions of output tensor

public:
    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param tr Tensor transformation.
     **/
    to_copy(dense_tensor_rd_i<N, T> &ta,
            const tensor_transf_t &tr = tensor_transf_t());

    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient.
     **/
    to_copy(dense_tensor_rd_i<N, T> &ta, T c);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor indexes.
        \param c Coefficient.
     **/
    to_copy(dense_tensor_rd_i<N, T> &ta, const permutation<N> &p,
        T c = 1.0);


    /** \brief Virtual destructor
     **/
    virtual ~to_copy() { }

    /** \brief Prefetches the source tensor
     **/
    void prefetch();

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param c Scaling coefficient
        \param tb Output tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<N, T> &tb);

    //@}

private:
    /** \brief Creates the dimensions of the output using an input
            tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(dense_tensor_rd_i<N, T> &ta,
        const permutation<N> &perm);
};


template<size_t N>
using tod_copy = to_copy<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_TO_COPY_H
