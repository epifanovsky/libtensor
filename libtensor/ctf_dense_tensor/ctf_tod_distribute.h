#ifndef LIBTENSOR_CTF_TOD_DISTRIBUTE_H
#define LIBTENSOR_CTF_TOD_DISTRIBUTE_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Copies a local dense tensor to a distributed CTF tensor
    \tparam N Tensor order.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_distribute : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_t; //!< Local dense tensor

public:
    /** \brief Initializes the operation
        \param t Local dense tensor.
     **/
    ctf_tod_distribute(dense_tensor_rd_i<N, double> &t) : m_t(t) { }

    /** \brief Writes the data to a distributed tensor
        \param dt Distributed tensor.
     **/
    void perform(ctf_dense_tensor_i<N, double> &dt);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_DISTRIBUTE_H

