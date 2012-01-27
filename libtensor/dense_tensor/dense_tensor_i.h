#ifndef LIBTENSOR_DENSE_TENSOR_I_H
#define LIBTENSOR_DENSE_TENSOR_I_H

#include <libtensor/core/dimensions.h>

namespace libtensor {


template<size_t N, typename T> class dense_tensor_base_ctrl;
template<size_t N, typename T> class dense_tensor_rd_ctrl;
template<size_t N, typename T> class dense_tensor_wr_ctrl;


/** \brief Dense tensor interface (abstract base class)
    \tparam N Tensor order.
    \tparam T Tensor element type.

    <b>Access to tensors</b>

    This abstract base class contains methods that are common to both
    read-only and write-only interfaces.

    Tensors shall not be accessed directly using these interfaces (hence all
    the access methods are protected). Control classes should be used to get
    access to tensor data. This is done to create a layer of security and
    prevent data corruption in the case of an exception.

    <b>Data format</b>

    Raw tensor data is stored in a vectorized form. The first index of a tensor
    is the slowest-running index, and the last index is the fastest-running
    index. The elements of the array are sorted in the order of the running
    index. In the case of a order-two tensor (matrix), this type of
    representation is called the row-major order (C-type).

    \sa dense_tensor_rd_i, dense_tensor_wr_i, dense_tensor_i,
        dense_tensor_base_ctrl

    \ingroup libtensor_dense_tensor
 **/
template<size_t N, typename T>
class dense_tensor_base_i {
    friend class dense_tensor_base_ctrl<N, T>;

public:
    typedef unsigned long session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~dense_tensor_base_i() { }

    /** \brief Returns the dimensions of the tensor
     **/
    virtual const dimensions<N> &get_dims() const = 0;

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests to open a session with the tensor
        \return Session handle.
     **/
    virtual session_handle_type on_req_open_session() = 0;

    /** \brief Handles requests to close a previously opened session
        \param h Session handle.
     **/
    virtual void on_req_close_session(const session_handle_type &h) = 0;

    /** \brief Handles requests to prefetch tensor data
        \param h Session handle.
     **/
    virtual void on_req_prefetch(const session_handle_type &h) = 0;


    //@}
};


/** \brief Read-only interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class contains methods that provide read-only access
    to a dense tensor.

    \sa dense_tensor_base_i, dense_tensor_wr_i, dense_tensor_i,
        dense_tensor_rd_ctrl

    \ingroup libtensor_dense_tensor
 **/
template<size_t N, typename T>
class dense_tensor_rd_i : virtual public dense_tensor_base_i<N, T> {
    friend class dense_tensor_rd_ctrl<N, T>;

public:
    typedef typename dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~dense_tensor_rd_i() { }

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests to provide a constant physical pointer
            to tensor data
        \param h Session handle.
     **/
    virtual const T *on_req_const_dataptr(const session_handle_type &h) = 0;

    /** \brief Handles requests to return a constant physical pointer to tensor
            data
        \param h Session handle.
        \param p Data pointer previously obtained from on_req_const_dataptr()
     **/
    virtual void on_ret_const_dataptr(const session_handle_type &h,
        const T *p) = 0;

    //@}

};



/** \brief Write-only interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class contains methods that provide write-only access
    to a dense tensor.

    \sa dense_tensor_base_i, dense_tensor_rd_i, dense_tensor_i,
        dense_tensor_wr_ctrl

    \ingroup libtensor_dense_tensor
 **/
template<size_t N, typename T>
class dense_tensor_wr_i : virtual public dense_tensor_base_i<N, T> {
    friend class dense_tensor_wr_ctrl<N, T>;

public:
    typedef typename dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~dense_tensor_wr_i() { }

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests to provide a physical pointer to tensor data
        \param h Session handle.
     **/
    virtual T *on_req_dataptr(const session_handle_type &h) = 0;

    /** \brief Handles requests to return a physical pointer to tensor data
        \param h Session handle.
        \param p Data pointer previously obtained from on_req_dataptr()
     **/
    virtual void on_ret_dataptr(const session_handle_type &h, const T *p) = 0;

    //@}

};



/** \brief Read-write interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class combines dense_tensor_rd_i and dense_tensor_wr_i.

    \sa dense_tensor_base_i, dense_tensor_rd_i, dense_tensor_wr_i,
        dense_tensor_ctrl

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class dense_tensor_i :
    public dense_tensor_rd_i<N, T>, public dense_tensor_wr_i<N, T> {

public:
    typedef typename dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~dense_tensor_i() { }

};


} // namespace libtensor

#endif // LIBTENSOR_DENSE_TENSOR_I_H
