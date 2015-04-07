#ifndef LIBTENSOR_CTF_DENSE_TENSOR_I_H
#define LIBTENSOR_CTF_DENSE_TENSOR_I_H

#include <libtensor/core/dimensions.h>
#include "ctf.h"
#include "ctf_symmetry.h"

namespace libtensor {


template<size_t N, typename T> class ctf_dense_tensor_ctrl;


/** \brief CTF tensor interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    <b>Access to tensors</b>

    \sa ctf_dense_tensor_ctrl

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, typename T>
class ctf_dense_tensor_i {
    friend class ctf_dense_tensor_ctrl<N, T>;

public:
    /** \brief Virtual destructor
     **/
    virtual ~ctf_dense_tensor_i() { }

    /** \brief Returns the dimensions of the tensor
     **/
    virtual const dimensions<N> &get_dims() const = 0;

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests for the CTF tensor object
        \return CTF tensor object
     **/
    virtual tCTF_Tensor<T> &on_req_ctf_tensor() = 0;

    /** \brief Handles requests for the symmetry of the CTF tensor
     **/
    virtual const ctf_symmetry<N, T> &on_req_symmetry() = 0;

    /** \brief Handles requests to reset the symmetry of the CTF tensor
     **/
    virtual void on_reset_symmetry(const ctf_symmetry<N, T> &sym) = 0;

    /** \brief Handles requests to adjust the symmetry of the CTF tensor
     **/
    virtual void on_adjust_symmetry(const ctf_symmetry<N, T> &sym) = 0;

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_I_H
