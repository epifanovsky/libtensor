#ifndef LIBTENSOR_MP_SAFE_TENSOR_H
#define LIBTENSOR_MP_SAFE_TENSOR_H

#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/inst/dense_tensor_impl.h>
#include "default_sync_policy.h"
#include "mp_safe_tensor_lock.h"

namespace libtensor {


/** \brief Thread-safe %tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Allocator.
    \tparam Sync Synchronization policy.

    \ingroup libtensor_mp
 **/
template<size_t N, typename T, typename Alloc,
    typename Sync = default_sync_policy>
class mp_safe_tensor : public dense_tensor<N, T, Alloc> {
public:
    typedef typename dense_tensor_i<N, T>::session_handle_type
        handle_t; //!< Session handle type

private:
    size_t m_count; //!< Session count
    mutex *m_mtx; //!< Mutex lock

public:
    mp_safe_tensor(const dimensions<N> &dims) :
        dense_tensor<N, T, Alloc>(dims), m_count(0), m_mtx(0) { }

    virtual ~mp_safe_tensor() {

        mp_safe_tensor_lock::get_instance().lock();
        delete m_mtx;
        mp_safe_tensor_lock::get_instance().unlock();
    }

protected:
    virtual handle_t on_req_open_session() {

        mp_safe_tensor_lock::get_instance().lock();
        m_count++;
        if(m_mtx == 0) m_mtx = new mutex;
        mp_safe_tensor_lock::get_instance().unlock();

        auto_lock<mutex> lock(*m_mtx);
        return dense_tensor<N, T, Alloc>::on_req_open_session();
    }

    virtual void on_req_close_session(const handle_t &h) {

        {
            auto_lock<mutex> lock(*m_mtx);
            dense_tensor<N, T, Alloc>::on_req_close_session(h);
        }

        mp_safe_tensor_lock::get_instance().lock();
        if(--m_count == 0) {
            delete m_mtx; m_mtx = 0;
        }
        mp_safe_tensor_lock::get_instance().unlock();
    }

    virtual void on_req_prefetch(const handle_t &h) {

        auto_lock<mutex> lock(*m_mtx);
        dense_tensor<N, T, Alloc>::on_req_prefetch(h);
    }

    virtual T *on_req_dataptr(const handle_t &h) {

        auto_lock<mutex> lock(*m_mtx);
        return dense_tensor<N, T, Alloc>::on_req_dataptr(h);
    }

    virtual void on_ret_dataptr(const handle_t &h, const T *p) {

        auto_lock<mutex> lock(*m_mtx);
        dense_tensor<N, T, Alloc>::on_ret_dataptr(h, p);
    }

    virtual const T *on_req_const_dataptr(const handle_t &h) {

        auto_lock<mutex> lock(*m_mtx);
        return dense_tensor<N, T, Alloc>::on_req_const_dataptr(h);
    }

    virtual void on_ret_const_dataptr(const handle_t &h, const T *p) {

        auto_lock<mutex> lock(*m_mtx);
        dense_tensor<N, T, Alloc>::on_ret_const_dataptr(h, p);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_MP_SAFE_TENSOR_H
