#ifndef LIBTENSOR_CTF_TOD_COLLECT_H
#define LIBTENSOR_CTF_TOD_COLLECT_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Copies a distributed CTF tensor to a local dense tensor
    \tparam N Tensor order.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_collect : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_dense_tensor_i<N, double> &m_dt; //!< Distributed CTF tensor

public:
    /** \brief Initializes the operation
        \param dt Distributed CTF tensor.
     **/
    ctf_tod_collect(ctf_dense_tensor_i<N, double> &dt) : m_dt(dt) { }

    /** \brief Collects the data to a local dense tensor
        \param t Dense tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_COLLECT_H

