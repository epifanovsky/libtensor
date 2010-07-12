#ifndef LIBTENSOR_MP_SAFE_TENSOR_H
#define LIBTENSOR_MP_SAFE_TENSOR_H

#include "../core/tensor.h"
#include "default_sync_policy.h"
#include "mp_safe_tensor_lock.h"

namespace libtensor {


/**	\brief Thread-safe %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Allocator.
	\tparam Sync Synchronization policy.

	\ingroup libtensor_mp
 **/
template<size_t N, typename T, typename Alloc,
	typename Sync = default_sync_policy>
class mp_safe_tensor : public tensor<N, T, Alloc> {
public:
	typedef typename tensor_i<N, T>::handle_t
		handle_t; //!< Session handle type

private:
	typedef typename Sync::mutex_t mutex_t; //!< Mutex type

private:
	size_t m_count; //!< Session count
	mutex_t *m_mtx; //!< Mutex lock

public:
	mp_safe_tensor(const dimensions<N> &dims) :
		tensor<N, T, Alloc>(dims), m_count(0), m_mtx(0) { }

	virtual ~mp_safe_tensor() {

		mp_safe_tensor_lock::get_instance().lock();
		delete m_mtx;
		mp_safe_tensor_lock::get_instance().unlock();
	}

protected:
	virtual handle_t on_req_open_session() {

		mp_safe_tensor_lock::get_instance().lock();
		m_count++;
		if(m_mtx == 0) m_mtx = new mutex_t;
		mp_safe_tensor_lock::get_instance().unlock();

		m_mtx->lock();
		try {
			handle_t h = tensor<N, T, Alloc>::on_req_open_session();
			m_mtx->unlock();
			return h;
		} catch(...) {
			m_mtx->unlock();
			throw;
		}
	}

	virtual void on_req_close_session(const handle_t &h) {

		mp_safe_tensor_lock::get_instance().lock();
		if(m_count > 0) m_count--;
		mp_safe_tensor_lock::get_instance().unlock();

		m_mtx->lock();
		try {
			tensor<N, T, Alloc>::on_req_close_session(h);
		} catch(...) {
			m_mtx->unlock();
			throw;
		}
		m_mtx->unlock();

		mp_safe_tensor_lock::get_instance().lock();
		if(m_count == 0) {
			delete m_mtx; m_mtx = 0;
		}
		mp_safe_tensor_lock::get_instance().unlock();
	}

	virtual void on_req_prefetch(const handle_t &h) {

		m_mtx->lock();
		tensor<N, T, Alloc>::on_req_prefetch(h);
		m_mtx->unlock();
	}

	virtual T *on_req_dataptr(const handle_t &h) {

		m_mtx->lock();
		T *p = tensor<N, T, Alloc>::on_req_dataptr(h);
		m_mtx->unlock();
		return p;
	}

	virtual void on_ret_dataptr(const handle_t &h, const T *p) {

		m_mtx->lock();
		tensor<N, T, Alloc>::on_ret_dataptr(h, p);
		m_mtx->unlock();
	}

	virtual const T *on_req_const_dataptr(const handle_t &h) {

		m_mtx->lock();
		const T *p = tensor<N, T, Alloc>::on_req_const_dataptr(h);
		m_mtx->unlock();
		return p;
	}

	virtual void on_ret_const_dataptr(const handle_t &h, const T *p) {

		m_mtx->lock();
		tensor<N, T, Alloc>::on_ret_const_dataptr(h, p);
		m_mtx->unlock();
	}

};


} // namespace libtensor

#endif // LIBTENSOR_MP_SAFE_TENSOR_H
