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

    /** \brief Requests the symmetry of the CTF tensor
     **/
    const ctf_symmetry<N, T> &req_symmetry() {
        return m_t.on_req_symmetry();
    }

    /** \brief Requests the CTF tensor object
        \param icomp Symmetry component
     **/
    CTF::Tensor<T> &req_ctf_tensor(size_t icomp = 0) {
        return m_t.on_req_ctf_tensor(icomp);
    }

    /** \brief Resets the symmetry of the CTF tensor
     **/
    void reset_symmetry(const ctf_symmetry<N, T> &sym) {
        m_t.on_reset_symmetry(sym);
    }

    /** \brief Adjusts the symmetry of the CTF tensor
     **/
    void adjust_symmetry(const ctf_symmetry<N, T> &sym) {
        m_t.on_adjust_symmetry(sym);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_CTRL_H
