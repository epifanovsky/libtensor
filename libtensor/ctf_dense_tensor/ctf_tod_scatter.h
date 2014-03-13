#ifndef LIBTENSOR_CTF_TOD_SCATTER_H
#define LIBTENSOR_CTF_TOD_SCATTER_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Scatters a lower-order tensor in a higher-order tensor
    \tparam N Order of the first tensor.
    \tparam M Order of the result less the order of the first tensor.

    \sa tod_scatter

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M>
class ctf_tod_scatter : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NC = N + M
    };

private:
    ctf_dense_tensor_i<NA, double> &m_ta; //!< First tensor (A)
    tensor_transf<NC, double> m_trc; //!< Transformation of result (C)

public:
    /** \brief Initializes the operation
        \param ta Input tensor
        \param trc Tensor transformation applied to result
     **/
    ctf_tod_scatter(
        ctf_dense_tensor_i<NA, double> &ta,
        const tensor_transf<NC, double> &trc = tensor_transf<NC, double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_scatter() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<NC, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SCATTER_H
