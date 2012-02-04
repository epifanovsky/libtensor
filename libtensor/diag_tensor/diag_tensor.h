#ifndef LIBTENSOR_DIAG_TENSOR_H
#define LIBTENSOR_DIAG_TENSOR_H

#include <map>
#include "diag_tensor_i.h"
#include "diag_tensor_space.h"

namespace libtensor {


/** \brief Implementation of general diagonal tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, typename T, typename Alloc>
class diag_tensor : public diag_tensor_i<N, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename diag_tensor_base_i<N, T>::session_handle_type
        session_handle_type; //!< Session handle type

private:
    struct pointer_record {
        typename Alloc::pointer_type vptr; // !< Virtual pointer
        T *dataptr; //!< Physical data pointer
        size_t ptrcnt; //!< Pointer count
        const T *const_dataptr; //!< Physical const data pointer
        size_t const_ptrcnt; //!< Const pointer count
        pointer_record() : vptr(Alloc::invalid_pointer), dataptr(0), ptrcnt(0),
            const_dataptr(0), const_ptrcnt(0) { }
    };

private:
    diag_tensor_space<N> m_spc; //!< Tensor space
    std::map<size_t, pointer_record> m_ptr; //!< Data pointers

public:
    /** \brief Initializes the tensor
        \param spc Tensor space.
     **/
    diag_tensor(const diag_tensor_space<N> &spc);

    /** \brief Virtual destructor
     **/
    virtual ~diag_tensor();

    /** \brief Returns the dimensions of the tensor
     **/
    virtual const dimensions<N> &get_dims() const {
        return m_spc.get_dims();
    }

    /** \brief Returns the space of the tensor
     **/
    virtual const diag_tensor_space<N> &get_space() const {
        return m_spc;
    }

protected:
    //! \name Implementation of diag_tensor_base_i
    //@{

    /** \brief Request to open a session with the tensor
        \return Session handle.
     **/
    virtual session_handle_type on_req_open_session();

    /** \brief Request to close a previously opened session
        \param h Session handle.
     **/
    virtual void on_req_close_session(const session_handle_type &h);

    //@}


    //! \name Implementation of diag_tensor_rd_i
    //@{

    /** \brief Requests (checks out) read-only data pointer
        \param h Session handle.
        \param n Subspace number.
     **/
    virtual const T *on_req_const_dataptr(const session_handle_type &h,
        size_t ssn);

    /** \brief Returns (checks in) read-only data pointer
        \param h Session handle.
        \param n Subspace number.
        \param p Data pointer.
     **/
    virtual void on_ret_const_dataptr(const session_handle_type &h,
        size_t ssn, const T *p);

    //@}


    //! \name Implementation of diag_tensor_wr_i
    //@{

    /** \brief Requests (checks out) read-write data pointer
        \param h Session handle.
        \param n Subspace number.
     **/
    virtual T *on_req_dataptr(const session_handle_type &h, size_t ssn);

    /** \brief Returns (checks in) read-write data pointer
        \param h Session handle.
        \param n Subspace number.
        \param p Data pointer.
     **/
    virtual void on_ret_dataptr(const session_handle_type &h, size_t ssn, T *p);

    //@}

private:
    /** \brief Private copy constructor
     **/
    diag_tensor(const diag_tensor&);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TENSOR_H

