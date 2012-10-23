#ifndef LIBTENSOR_TOD_EXTRACT_H
#define LIBTENSOR_TOD_EXTRACT_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/index.h>
#include <libtensor/core/mask.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Extracts a tensor with a smaller dimension from a %tensor
    \tparam N Tensor order.
    \tparam M Number of fixed indexes in the initial tensor.

    Extracts a tensor with a smaller dimension from a %tensor. The
    indexes which remain intact is specified by a %mask, unmasked indexes
    are remain constant. The order of the result is (n-m), where n is
    the order of the original %tensor, m is the number of constant indexes.

    The order of indexes in the result is the same as in the argument.
    The example of use:
    \f[ c_i = a_{ij} \qquad c_{ip} = a_{ijpc} \qquad c_{ip} = a_{ijcp} \f]
    The specified permutation may be applied to the result to alter the
    order of the indexes.

    A coefficient (default 1.0) is specified to scale the elements along
    with the extraction of the tensor.

    If the number of set bits in the %mask is not equal to M, the %mask
    is incorrect, which causes a \c bad_parameter exception upon the
    creation of the operation. If the %dimensions of the output %tensor
    are wrong, the \c bad_dimensions exception is thrown.

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_extract : public timings< tod_extract<N, M> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N, //!< Order of the source %tensor
        NB = N - M //!< Order of the destination %tensor
    };

public:
    typedef tensor_transf<NB, double> tensor_transf_type;

private:
    dense_tensor_rd_i<NA, double> &m_t; //!< Input %tensor
    mask<NA> m_mask; //!< Mask for extraction
    permutation<NB> m_perm; //!< Permutation of the result
    double m_c; //!< Scaling coefficient
    dimensions<NB> m_dims; //!< Dimensions of the result
    index<NA> m_idx;//!< Index for extraction

public:
    /** \brief Creates the operation
        \param t Input tensor.
        \param m Extraction mask.
        \param idx Index for extraction.
        \param tr Transformation of result (default \f$(P_0,1.0)\f$).
    **/
    tod_extract(dense_tensor_rd_i<NA, double> &t,
            const mask<NA> &m, const index<NA> &idx,
            const tensor_transf_type &tr = tensor_transf_type());

    /** \brief Creates the operation
        \param t Input tensor.
        \param m Extraction mask.
        \param idx Index for extraction.
        \param c Scaling coefficient.
    **/
    tod_extract(dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
            const index<NA> &idx, double c);

    /** \brief Creates the operation
        \param t Input tensor.
        \param m Extraction mask.
        \param idx Index for extraction.
        \param p Permutation of result.
        \param c Scaling coefficient (default 1.0)
    **/
    tod_extract(dense_tensor_rd_i<NA, double> &t, const mask<NA> &m,
            const index<NA> &idx, const permutation<NB> &p, double c = 1.0);

    /** \brief Performs the operation
        \param zero Zero output first
        \param tb Output tensor.
    **/
    void perform(bool zero, dense_tensor_wr_i<NB, double> &tb);

private:
    /** \brief Forms the %dimensions of the output or throws an
        exception if the input is incorrect
    **/
    static dimensions<N - M> mk_dims(
        const dimensions<NA> &dims, const mask<NA> &msk);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_EXTRACT_H
