#ifndef LIBTENSOR_CTF_TOD_SET_H
#define LIBTENSOR_CTF_TOD_SET_H

#include <libtensor/core/noncopyable.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Sets all elements of a distributed tensor to a given value
    \tparam N Tensor order.

    \sa tod_set

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_set : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    ctf_tod_set(double v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_H
