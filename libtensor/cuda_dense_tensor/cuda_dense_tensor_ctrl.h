#ifndef LIBTENSOR_DENSE_CUDA_TENSOR_CTRL_H
#define LIBTENSOR_DENSE_CUDA_TENSOR_CTRL_H

#include "cuda_dense_tensor_i.h"
#include <libtensor/cuda/cuda_pointer.h>

namespace libtensor {


/** \brief CUDA Dense tensor control
    \tparam N Tensor order.
    \tparam T Tensor element type.

    Tensor controls provide methods to read the tensor's data. This indirect
    way of accessing tensors is needed to prevent tensor corruption in
    the case of an exception.

    Upon creation, this control opens a session with the given tensor. When
    destroyed, it closes the session. The session remains open during
    the life time of the control object.

    There are two types of controls, one for read-only access, and the other for
    write-only access.

    \sa cuda_dense_tensor_rd_ctrl, cuda_dense_tensor_wr_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_base_ctrl {
public:
    typedef typename cuda_dense_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

private:
    cuda_dense_tensor_base_i<N, T> &m_t; //!< Dense tensor object
    session_handle_type m_h; //!< Session handle

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the control object, initiates a session
        \param t Tensor instance.
     **/
    cuda_dense_tensor_base_ctrl(cuda_dense_tensor_base_i<N, T> &t) : m_t(t) {
        m_h = m_t.on_req_open_session();
    }

    /** \brief Destroys the control object, closes the session
     **/
    virtual ~cuda_dense_tensor_base_ctrl() {
        m_t.on_req_close_session(m_h);
    }

    //@}

    //! \name Tensor access methods
    //@{

    /** \brief Informs the tensor object that the data will soon be required
            so it can be fetched from external storage to local memory
     **/
    void req_prefetch() {
        m_t.on_req_prefetch(m_h);
    }

    /** \brief Asks the VM subsystem to keep the data in core memory
        \param pri Set (true) or unset (false) VM in-core priority.
     **/
    void req_priority(bool pri) {
        m_t.on_req_priority(m_h, pri);
    }

    //@}

protected:
    /** \brief Returns the session handle
     **/
    const session_handle_type &get_h() const {
        return m_h;
    }

};


/** \brief Read-only dense tensor control
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This control provides read-only access to raw tensor data.

    \sa cuda_dense_tensor_base_ctrl, cuda_dense_tensor_wr_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_rd_ctrl : virtual public cuda_dense_tensor_base_ctrl<N, T> {
private:
    cuda_dense_tensor_rd_i<N, T> &m_t; //!< Dense tensor object

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the control object
        \param t Tensor instance.
     **/
    cuda_dense_tensor_rd_ctrl(cuda_dense_tensor_rd_i<N, T> &t) :
        cuda_dense_tensor_base_ctrl<N, T>(t), m_t(t) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_rd_ctrl() { }

    //@}

    //! \name Tensor access methods
    //@{

    /** \brief Requests a read-only raw data pointer to tensor data and returns
            it to the user
     **/
    const cuda_pointer<const T> req_const_dataptr() {
        return m_t.on_req_const_dataptr(get_h());
    }

    /** \brief Checks in a raw data pointer previously obtained through
            req_const_dataptr(). The pointer may not be used after this.
     **/
    void ret_const_dataptr(cuda_pointer<const T> p) {
        m_t.on_ret_const_dataptr(get_h(), p);
    }

    //@}

protected:
    using cuda_dense_tensor_base_ctrl<N, T>::get_h;

};


/** \brief Write-only dense tensor control
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This control provides write-only access to raw tensor data.

    \sa cuda_dense_tensor_base_ctrl, cuda_dense_tensor_rd_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_wr_ctrl : virtual public cuda_dense_tensor_base_ctrl<N, T> {
private:
    cuda_dense_tensor_wr_i<N, T> &m_t; //!< Dense tensor object

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the control object
        \param t Tensor instance.
     **/
    cuda_dense_tensor_wr_ctrl(cuda_dense_tensor_wr_i<N, T> &t) :
        cuda_dense_tensor_base_ctrl<N, T>(t), m_t(t) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_wr_ctrl() { }

    //@}


    //! \name Tensor access methods
    //@{

    /** \brief Requests a write-only raw data pointer to tensor data and returns
            it to the user
     **/
    cuda_pointer<T> req_dataptr() {
        return m_t.on_req_dataptr(get_h());
    }

    /** \brief Checks in a raw data pointer previously obtained through
            req_dataptr(). The pointer may not be used after this.
     **/
    void ret_dataptr(cuda_pointer<T> p) {
        m_t.on_ret_dataptr(get_h(), p);
    }

    //@}

protected:
    using cuda_dense_tensor_base_ctrl<N, T>::get_h;

};


/** \brief Read-write dense tensor control
    \tparam N Tensor order.
    \tparam T Tensor element type.

    This control provides read-write access to raw tensor data.

    \sa cuda_dense_tensor_base_ctrl, cuda_dense_tensor_rd_ctrl, cuda_dense_tensor_wr_ctrl

    \ingroup libtensor_cuda_dense_tensor
 **/
template<size_t N, typename T>
class cuda_dense_tensor_ctrl :
    public cuda_dense_tensor_rd_ctrl<N, T>, public cuda_dense_tensor_wr_ctrl<N, T> {

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the control object
        \param t Tensor instance.
     **/
    cuda_dense_tensor_ctrl(cuda_dense_tensor_i<N, T> &t) :
        cuda_dense_tensor_base_ctrl<N, T>(t),
        cuda_dense_tensor_rd_ctrl<N, T>(t),  cuda_dense_tensor_wr_ctrl<N, T>(t) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~cuda_dense_tensor_ctrl() { }

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_DENSE_CUDA_TENSOR_CTRL_H
