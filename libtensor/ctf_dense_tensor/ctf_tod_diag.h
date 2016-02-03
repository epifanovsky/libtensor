#ifndef LIBTENSOR_CTF_TOD_DIAG_H
#define LIBTENSOR_CTF_TOD_DIAG_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Extracts a generalized diagonal from a distributed tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \sa tod_diag

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M>
class ctf_tod_diag : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NB = M
    };

private:
    ctf_dense_tensor_i<NA, double> &m_ta; //!< Source tensor
    sequence<NA, size_t> m_mask; //!< Diagonal mask
    tensor_transf<NB, double> m_trb; //!< Transformation of the result
    dimensions<NB> m_dimsb; //!< Dimensions of the result

public:
    /** \brief Creates the operation
        \param ta Input tensor.
        \param m Diagonal mask.
        \param trbTransformation of the result.
     **/
    ctf_tod_diag(
        ctf_dense_tensor_i<NA, double> &ta,
        const sequence<NA, size_t> &m,
        const tensor_transf<NB, double> &trb = tensor_transf<NB, double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_diag() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tb Output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<NB, double> &tb);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DIAG_H
