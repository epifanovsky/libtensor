#ifndef LIBTENSOR_CTF_TOD_SET_DIAG_H
#define LIBTENSOR_CTF_TOD_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Assigns the diagonal elements of a distributed tensor to a value
    \tparam N Tensor order.

    \sa tod_set_diag

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_set_diag : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Tensor element value (default 0.0).
     **/
    ctf_tod_set_diag(double v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
     **/
    void perform(ctf_dense_tensor_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_DIAG_H
