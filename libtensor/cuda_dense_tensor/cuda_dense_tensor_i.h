#ifndef LIBTENSOR_CUDA_DENSE_TENSOR_I_H
#define LIBTENSOR_CUDA_DENSE_TENSOR_I_H

#include <libtensor/core/dimensions.h>
#include <libtensor/cuda/cuda_pointer.h>

namespace libtensor {


template<size_t N, typename T> class cuda_dense_tensor_base_ctrl;
template<size_t N, typename T> class cuda_dense_tensor_rd_ctrl;
template<size_t N, typename T> class cuda_dense_tensor_wr_ctrl;


/** \brief CUDA Dense tensor interface (abstract base class)
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

    \sa cuda_dense_tensor_rd_i, cuda_dense_tensor_wr_i, cuda_dense_tensor_i,
        cuda_dense_tensor_base_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_base_i {
    friend class cuda_dense_tensor_base_ctrl<N, T>;

public:
    typedef unsigned long session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_base_i() { }

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

    /** \brief Handles requests to set in-core VM priority
        \param h Session handle.
        \param pri Set (true)/unset (false) priority.
     **/
    virtual void on_req_priority(const session_handle_type &h, bool pri) = 0;

    //@}
};


/** \brief Read-only interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class contains methods that provide read-only access
    to a dense tensor.

    \sa cuda_dense_tensor_base_i, cuda_dense_tensor_wr_i, cuda_dense_tensor_i,
        cuda_dense_tensor_rd_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_rd_i : virtual public cuda_dense_tensor_base_i<N, T> {
    friend class cuda_dense_tensor_rd_ctrl<N, T>;

public:
    typedef typename cuda_dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_rd_i() { }

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests to provide a constant CUDA pointer
            to tensor data
        \param h Session handle.
     **/
    virtual cuda_pointer<const T> on_req_const_dataptr(const session_handle_type &h) = 0;

    /** \brief Handles requests to return a constant physical pointer to tensor
            data
        \param h Session handle.
        \param p Data pointer previously obtained from on_req_const_dataptr()
     **/
    virtual void on_ret_const_dataptr(const session_handle_type &h,
    		cuda_pointer<const T> p) = 0;

    //@}

};



/** \brief Write-only interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class contains methods that provide write-only access
    to a dense tensor.

    \sa cuda_dense_tensor_base_i, cuda_dense_tensor_rd_i, cuda_dense_tensor_i,
        cuda_dense_tensor_wr_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_wr_i : virtual public cuda_dense_tensor_base_i<N, T> {
    friend class cuda_dense_tensor_wr_ctrl<N, T>;

public:
    typedef typename cuda_dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_wr_i() { }

protected:
    //! \name Tensor events
    //@{

    /** \brief Handles requests to provide a physical pointer to tensor data
        \param h Session handle.
     **/
    virtual cuda_pointer<T> on_req_dataptr(const session_handle_type &h) = 0;

    /** \brief Handles requests to return a physical pointer to tensor data
        \param h Session handle.
        \param p Data pointer previously obtained from on_req_dataptr()
     **/
    virtual void on_ret_dataptr(const session_handle_type &h, cuda_pointer<T> p) = 0;

    //@}

};



/** \brief Read-write interface to dense tensors
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This abstract base class combines cuda_dense_tensor_rd_i and cuda_dense_tensor_wr_i.

    \sa cuda_dense_tensor_base_i, cuda_dense_tensor_rd_i, cuda_dense_tensor_wr_i,
        cuda_dense_tensor_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_i :
    public cuda_dense_tensor_rd_i<N, T>, public cuda_dense_tensor_wr_i<N, T> {

public:
    typedef typename cuda_dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_i() { }

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_DENSE_TENSOR_I_H
