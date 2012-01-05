#ifndef LIBTENSOR_DENSE_TENSOR_I_H
#define LIBTENSOR_DENSE_TENSOR_I_H

#include <libtensor/core/dimensions.h>

namespace libtensor {


template<size_t N, typename T> class tensor_ctrl;


/** \brief Tensor interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    Interface for tensors that are represented as a multi-dimensional
    array of %tensor elements. The elements are arranged linearly in memory
    in the order of the running multi-dimensional %index such that the last
    entry of the %index is the fastest running.

    For example, a two-%index %tensor (matrix) would have its elements in
    the row-major order: (0,0); (0,1); ...; (0,n); (1,0); (1,1); ...; (m,n).

    The size of the %tensor is specified by its %dimensions -- the number
    of entries along each dimension of the array. The %dimensions are
    obtained using get_dims().

    While the internal format of the %tensor is not established by this
    interface, the physical pointer to the array shall contain dense data
    ordered as specified above.

    Raw %tensor data are accessed via a control object (tensor_ctrl), which
    provides user-end methods. Implementations of this interface, however,
    shall realize a set of protected functions that are necessary to handle
    requests from the user.

    The control object shall start by requesting to open a session with
    the %tensor. The request is handled by on_req_open_session(), which
    shall return a handle for the session. The handle shall be used later
    to identify the session. Session termination requests are handled by
    on_req_close_session(), which shall invalidate the handle.

    \sa tensor_ctrl

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class dense_tensor_i {
    friend class tensor_ctrl<N, T>;

public:
    typedef size_t handle_t; //!< Session handle type

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Virtual destructor
     **/
    virtual ~dense_tensor_i() { }

    //@}


    //!	\name Tensor dimensions
    //@{

    /**	\brief Returns the %dimensions of the %tensor
     **/
    virtual const dimensions<N> &get_dims() const = 0;

    //@}


protected:
    //!	\name Tensor events
    //@{

    /**	\brief Handles requests to open a session with the %tensor
            \return Session handle.
     **/
    virtual handle_t on_req_open_session() = 0;

    /**	\brief Handles requests to close a previously opened session
            \param h Session handle.
     **/
    virtual void on_req_close_session(const handle_t &h) = 0;

    /**	\brief Handles requests to prefetch %tensor data
            \param h Session handle.
     **/
    virtual void on_req_prefetch(const handle_t &h) = 0;

    /**	\brief Handles requests to provide a physical pointer to
                    %tensor data
            \param h Session handle.
     **/
    virtual T *on_req_dataptr(const handle_t &h) = 0;

    /**	\brief Handles requests to return the physical pointer to
                    %tensor data
            \param h Session handle.
            \param p Data pointer previously obtained from on_req_dataptr()
     **/
    virtual void on_ret_dataptr(const handle_t &h, const T *p) = 0;

    /**	\brief Handles requests to provide a constant physical pointer
                    to %tensor data
            \param h Session handle.
     **/
    virtual const T *on_req_const_dataptr(const handle_t &h) = 0;

    /**	\brief Handles requests to return the constant physical pointer
                    to %tensor data
            \param h Session handle.
            \param p Data pointer previously obtained from
                    on_req_const_dataptr()
     **/
    virtual void on_ret_const_dataptr(const handle_t &h, const T *p) = 0;

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_DENSE_TENSOR_I_H
