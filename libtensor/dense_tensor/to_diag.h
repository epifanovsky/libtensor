#ifndef LIBTENSOR_TO_DIAG_H
#define LIBTENSOR_TO_DIAG_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/mask.h>
#include <libtensor/core/permutation.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Extracts a general diagonal from a %tensor
    \tparam N Tensor order.
    \tparam M Result order.

    Extracts a general multi-dimensional diagonal from a %tensor. The
    diagonal(s) to extract is specified by a sequence object, which serves as
    a %mask. Dimensions for which the sequence is 0 are unmasked and remain
    intact. All dimensions for which the sequence has the same non-zero value
    are collapsed into one. The order of the result is m, while n is the order
    of the original %tensor.

    The order of dimensions in the result is the number of argument with
    the exception of the collapsed diagonal(s). The dimension of any diagonal
    in the result correspond to the first of its dimensions in the argument,
    for example:
    \f[
        c_i = a_{ii}     \qquad c_{ip} = a_{iip}    \qquad
        c_{ip} = a_{ipi} \qquad c_{ija} = a_{ijiaj}
    \f]
    The specified permutation may be applied to the result to alter the
    order of the dimensions.

    A coefficient (default 1.0) is specified to scale the elements along
    with the extraction of the diagonal.

    If the number of unmasked dimensions plus the largest value in the
    sequence are not equal to M, the sequence is incorrect, which causes
    a \c bad_parameter exception upon the creation of the operation. If
    the %dimensions of the output %tensor are wrong, the \c bad_dimensions
    exception is thrown.

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M, typename T>
class to_diag : public timings< to_diag<N, M, T> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N, //!< Order of the source %tensor
        NB = M  //!< Order of the destination %tensor
    };

    typedef tensor_transf<NB, T> tensor_transf_type;

private:
    dense_tensor_rd_i<NA, T> &m_t; //!< Input %tensor
    sequence<NA, size_t> m_mask; //!< Diagonal mask
    permutation<NB> m_perm; //!< Permutation of the result
    T m_c; //!< Scaling coefficient
    dimensions<NB> m_dims; //!< Dimensions of the result

public:
    /** \brief Creates the operation
        \param t Input %tensor.
        \param m Diagonal mask.
        \param p Permutation of result.
        \param c Scaling coefficient (default 1.0)
     **/
    to_diag(dense_tensor_rd_i<NA, T> &t, const sequence<NA, size_t> &m,
            const tensor_transf_type &tr = tensor_transf_type());

    /** \brief Performs the operation, adds to the output
        \param zero Zero result first
        \param c Scalar transformation to apply before adding to result
        \param tb Output %tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<NB, T> &tb);

};

template<size_t N, size_t M>
using tod_diag = to_diag<N, M, double>;

} // namespace libtensor

#endif // LIBTENSOR_TO_DIAG_H
