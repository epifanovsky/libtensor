#ifndef LIBTENSOR_CTF_TOD_COPY_H
#define LIBTENSOR_CTF_TOD_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Copies the contents of a distributed tensor, permutes and scales
        the entries if necessary
    \tparam N Tensor order.

    \sa tod_copy

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_copy : public timings< ctf_tod_copy<N> >, public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_dense_tensor_i<N, double> &m_ta; //!< Source tensor
    tensor_transf<N, double> m_tra; //!< Transformation
    dimensions<N> m_dimsb; //!< Dimensions of output tensor

public:
    /** \brief Scaled copy
        \param ta Source tensor.
        \param c Coefficient (default 1.0).
     **/
    ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta, double c = 1.0);

    /** \brief Permuted and scaled copy
        \param ta Source tensor.
        \param perma Tensor permutation.
        \param c Coefficient (default 1.0).
     **/
    ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
        const permutation<N> &perma, double c = 1.0);

    /** \brief Transformed copy
        \param ta Source tensor.
        \param tra Tensor transformation.
     **/
    ctf_tod_copy(ctf_dense_tensor_i<N, double> &ta,
        const tensor_transf<N, double> &tra);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_copy() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tb Output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &tb);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_COPY_H
