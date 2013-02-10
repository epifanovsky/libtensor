#ifndef LIBTENSOR_CTF_DENSE_TENSOR_CTRL_H
#define LIBTENSOR_CTF_DENSE_TENSOR_CTRL_H

#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief CTF tensor control
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, typename T>
class ctf_dense_tensor_ctrl {
private:
    ctf_dense_tensor_i<N, T> &m_t; //!< CTF dense tensor object

public:
    /** \brief Initializes the control object
        \param t Tensor instance.
     **/
    ctf_dense_tensor_ctrl(ctf_dense_tensor_i<N, T> &t) : m_t(t) { }

    /** \brief Destroys the control object
     **/
    virtual ~ctf_dense_tensor_ctrl() { }

    /** \brief Requests tensor ID in CTF
     **/
    int req_tensor_id() {
        m_t.on_req_tensor_id();
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_CTRL_H
