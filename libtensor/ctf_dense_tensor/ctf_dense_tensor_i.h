#ifndef LIBTENSOR_CTF_DENSE_TENSOR_I_H
#define LIBTENSOR_CTF_DENSE_TENSOR_I_H

#include <libtensor/core/dimensions.h>

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

    /** \brief Handles requests for the tensor ID
        \return Tensor ID in CTF
     **/
    virtual int on_req_tensor_id() = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_I_H
