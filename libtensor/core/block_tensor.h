#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "../mp/default_sync_policy.h"
#include "block_index_space.h"
#include "block_map.h"
#include "block_tensor_i.h"
#include "immutable.h"
#include "orbit_list.h"
#include "tensor.h"
#include "tensor_ctrl.h"

namespace libtensor {

/**	\brief Block %tensor
	\tparam N Tensor order.
	\tparam T Tensor element type.
	\tparam Alloc Memory allocator.
	\tparam Sync Synchronization policy.

	\ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc,
	typename Sync = default_sync_policy>
class block_tensor : public block_tensor_i<N, T>, public immutable {
public:
	static const char *k_clazz; //!< Class name

private:
	typedef typename Sync::mutex_t mutex_t; //!< Mutex lock type
	typedef typename Sync::rwlock_t rwlock_t; //!< Read-write lock type

	class auto_lock {
	private:
		rwlock_t *m_lock;
		mutex_t *m_locku;
		bool m_wr;

	public:
		auto_lock(rwlock_t *lock, mutex_t *locku, bool wr) :
			m_lock(lock), m_locku(locku), m_wr(wr) {
			if(m_lock) {
				if(wr) {
					//m_locku->lock();
					m_lock->wrlock();
				} else {
					m_lock->rdlock();
				}
			}
		}

		~auto_lock() {
			if(m_lock) {
				m_lock->unlock();
				//if(m_wr) m_locku->unlock();
			}
		}

		void upgrade() {
			if(m_lock && !m_wr) {
				//m_locku->lock();
				m_lock->unlock();
				m_lock->wrlock();
				m_wr = true;
			}
		}

		void downgrade() {
			if(m_lock && m_wr) {
				m_lock->unlock();
				m_lock->rdlock();
				m_wr = false;
				//m_locku->unlock();
			}
		}

	};

public:
	block_index_space<N> m_bis; //!< Block %index space
	dimensions<N> m_bidims; //!< Block %index %dimensions
	symmetry<N, T> m_symmetry; //!< Block %tensor symmetry
	orbit_list<N, T> *m_orblst; //!< Orbit list
	bool m_orblst_dirty; //!< Whether the orbit list needs to be updated
	block_map<N, T, Alloc> m_map; //!< Block map
	block_map<N, T, Alloc> m_aux_map; //!< Auxiliary block map
	rwlock_t *m_lock; //!< Read-write lock
	mutex_t *m_locku; //!< Upgrade lock

public:
	//!	\name Construction and destruction
	//@{
	block_tensor(const block_index_space<N> &bis);
	block_tensor(const block_tensor<N, T, Alloc> &bt);
	virtual ~block_tensor();
	//@}

	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual const block_index_space<N> &get_bis() const;
	//@}

protected:
	//!	\name Implementation of libtensor::block_tensor_i<N, T>
	//@{
	virtual const symmetry<N, T> &on_req_const_symmetry() throw(exception);
	virtual symmetry<N, T> &on_req_symmetry() throw(exception);
	virtual tensor_i<N, T> &on_req_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_block(const index<N> &idx) throw(exception);
	virtual tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
		throw(exception);
	virtual void on_ret_aux_block(const index<N> &idx) throw(exception);
	virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_block(const index<N> &idx) throw(exception);
	virtual void on_req_zero_all_blocks() throw(exception);
	virtual void on_req_sync_on() throw(exception);
	virtual void on_req_sync_off() throw(exception);
	//@}

	//!	\name Implementation of libtensor::immutable
	//@{
	virtual void on_set_immutable();
	//@}

private:
	void update_orblst(auto_lock &lock);
};


template<size_t N, typename T, typename Alloc, typename Sync>
const char *block_tensor<N, T, Alloc, Sync>::k_clazz =
	"block_tensor<N, T, Alloc, Sync>";


template<size_t N, typename T, typename Alloc, typename Sync>
block_tensor<N, T, Alloc, Sync>::block_tensor(const block_index_space<N> &bis) :
	m_bis(bis),
	m_bidims(bis.get_block_index_dims()),
	m_symmetry(m_bis),
	m_orblst(0),
	m_orblst_dirty(true),
	m_lock(0), m_locku(0) {

}


template<size_t N, typename T, typename Alloc, typename Sync>
block_tensor<N, T, Alloc, Sync>::block_tensor(
	const block_tensor<N, T, Alloc> &bt) :

	m_bis(bt.get_bis()),
	m_bidims(bt.m_bidims),
	m_symmetry(bt.get_bis()),
	m_orblst(0),
	m_orblst_dirty(true),
	m_lock(0), m_locku(0) {

}


template<size_t N, typename T, typename Alloc, typename Sync>
block_tensor<N, T, Alloc, Sync>::~block_tensor() {

	delete m_lock;
	delete m_locku;
	delete m_orblst;
}


template<size_t N, typename T, typename Alloc, typename Sync>
const block_index_space<N> &block_tensor<N, T, Alloc, Sync>::get_bis()
	const {

	return m_bis;
}


template<size_t N, typename T, typename Alloc, typename Sync>
const symmetry<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_const_symmetry()
	throw(exception) {

	return m_symmetry;
}


template<size_t N, typename T, typename Alloc, typename Sync>
symmetry<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_symmetry()
	throw(exception) {

	static const char *method = "on_req_symmetry()";

	auto_lock lock(m_lock, m_locku, true);

	if(is_immutable()) {
		throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
			"symmetry");
	}
	m_orblst_dirty = true;

	return m_symmetry;
}


template<size_t N, typename T, typename Alloc, typename Sync>
tensor_i<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_block(const index<N>&)";

	auto_lock lock(m_lock, m_locku, false);

	update_orblst(lock);
	size_t absidx = m_bidims.abs_index(idx);
	if(!m_orblst->contains(absidx)) {
		throw symmetry_violation(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	if(!m_map.contains(absidx)) {
		lock.upgrade();
		dimensions<N> blkdims = m_bis.get_block_dims(idx);
		m_map.create(absidx, blkdims);
		lock.downgrade();
	}
	return m_map.get(absidx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_ret_block(const index<N> &idx)
	throw(exception) {

}


template<size_t N, typename T, typename Alloc, typename Sync>
tensor_i<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_aux_block(
	const index<N> &idx) throw(exception) {

	static const char *method = "on_req_aux_block(const index<N>&)";

	auto_lock lock(m_lock, m_locku, true);

	update_orblst(lock);
	size_t absidx = m_bidims.abs_index(idx);
	if(!m_orblst->contains(absidx)) {
		throw symmetry_violation(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	if(m_aux_map.contains(absidx)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Duplicate aux request.");
	}

	dimensions<N> blkdims = m_bis.get_block_dims(idx);
	m_aux_map.create(absidx, blkdims);

	return m_aux_map.get(absidx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_ret_aux_block(const index<N> &idx)
	throw(exception) {

	auto_lock lock(m_lock, m_locku, true);

	size_t absidx = m_bidims.abs_index(idx);
	m_aux_map.remove(absidx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
bool block_tensor<N, T, Alloc, Sync>::on_req_is_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_is_zero_block(const index<N>&)";

	auto_lock lock(m_lock, m_locku, false);

	update_orblst(lock);
	size_t absidx = m_bidims.abs_index(idx);
	if(!m_orblst->contains(absidx)) {
		throw symmetry_violation(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	return !m_map.contains(absidx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_zero_block(const index<N> &idx)
	throw(exception) {

	static const char *method = "on_req_zero_block(const index<N>&)";

	auto_lock lock(m_lock, m_locku, true);

	if(is_immutable()) {
		throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}
	update_orblst(lock);
	size_t absidx = m_bidims.abs_index(idx);
	if(!m_orblst->contains(absidx)) {
		throw symmetry_violation(g_ns, k_clazz, method,
			__FILE__, __LINE__,
			"Index does not correspond to a canonical block.");
	}
	m_map.remove(absidx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_zero_all_blocks()
	throw(exception) {

	static const char *method = "on_req_zero_all_blocks()";

	auto_lock lock(m_lock, m_locku, true);

	if(is_immutable()) {
		throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Immutable object cannot be modified.");
	}
	m_map.clear();
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_sync_on() throw(exception) {

	if(m_lock == 0) {
		m_lock = new rwlock_t;
		m_locku = new mutex_t;
	}
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_sync_off() throw(exception) {

	delete m_lock; m_lock = 0;
	delete m_locku; m_locku = 0;
}


template<size_t N, typename T, typename Alloc, typename Sync>
inline void block_tensor<N, T, Alloc, Sync>::on_set_immutable() {

	auto_lock lock(m_lock, m_locku, true);

	m_map.set_immutable();
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::update_orblst(auto_lock &lock) {

	if(m_orblst == 0 || m_orblst_dirty) {
		lock.upgrade();
		delete m_orblst;
		m_orblst = new orbit_list<N, T>(m_symmetry);
		m_orblst_dirty = false;
		lock.downgrade();
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H
