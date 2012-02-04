#ifndef LIBTENSOR_DIAG_TENSOR_CTRL_H
#define LIBTENSOR_DIAG_TENSOR_CTRL_H

#include "diag_tensor_i.h"

namespace libtensor {


template<size_t N, typename T>
class diag_tensor_base_ctrl {
public:
    typedef typename diag_tensor_base_i<N, T>::session_handle_type
        session_handle_type;

private:
    diag_tensor_base_i<N, T> &m_t; //!< Tensor object
    session_handle_type m_h; //!< Session handle

public:
    /** \brief Initializes the control object, initiates a session
        \param t Tensor.
     **/
    diag_tensor_base_ctrl(diag_tensor_base_i<N, T> &t) : m_t(t) {
        m_h = m_t.on_req_open_session();
    }

    /** \brief Destroys the control object, closes the session
     **/
    virtual ~diag_tensor_base_ctrl() {
        m_t.on_req_close_session(m_h);
    }

protected:
    /** \brief Returns the session handle
     **/
    const session_handle_type &get_h() const {
        return m_h;
    }

};


template<size_t N, typename T>
class diag_tensor_rd_ctrl : virtual public diag_tensor_base_ctrl<N, T> {
private:
    diag_tensor_rd_i<N, T> &m_t; //!< Tensor object

public:
    /** \brief Initializes the control object
        \param t Tensor.
     **/
    diag_tensor_rd_ctrl(diag_tensor_rd_i<N, T> &t) :
        diag_tensor_base_ctrl<N, T>(t), m_t(t) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~diag_tensor_rd_ctrl() { }

    /** \brief Requests a read-only raw data pointer to tensor data and returns
            it to the user
        \param ssn Subspace number.
     **/
    const T *req_const_dataptr(size_t ssn) {
        return m_t.on_req_const_dataptr(get_h(), ssn);
    }

    /** \brief Checks in a raw data pointer previously obtained through
            req_const_dataptr(). The pointer may not be used after this.
        \param ssn Subspace number.
        \param p Returned pointer.
     **/
    void ret_const_dataptr(size_t ssn, const T *p) {
        m_t.on_ret_const_dataptr(get_h(), ssn, p);
    }

protected:
    using diag_tensor_base_ctrl<N, T>::get_h;

};


template<size_t N, typename T>
class diag_tensor_wr_ctrl : virtual public diag_tensor_base_ctrl<N, T> {
private:
    diag_tensor_wr_i<N, T> &m_t; //!< Tensor object

public:
    /** \brief Initializes the control object
        \param t Tensor.
     **/
    diag_tensor_wr_ctrl(diag_tensor_wr_i<N, T> &t) :
        diag_tensor_base_ctrl<N, T>(t), m_t(t) {
    }

    /** \brief Virtual destructor
     **/
    virtual ~diag_tensor_wr_ctrl() { }

    /** \brief Requests a raw data pointer to tensor data and returns
            it to the user
        \param ssn Subspace number.
     **/
    T *req_dataptr(size_t ssn) {
        return m_t.on_req_dataptr(get_h(), ssn);
    }

    /** \brief Checks in a raw data pointer previously obtained through
            req_dataptr(). The pointer may not be used after this.
        \param ssn Subspace number.
        \param p Returned pointer.
     **/
    void ret_dataptr(size_t ssn, T *p) {
        m_t.on_ret_dataptr(get_h(), ssn, p);
    }

protected:
    using diag_tensor_base_ctrl<N, T>::get_h;

};



} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_CTRL_H

