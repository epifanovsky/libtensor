#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/immutable.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/mp/default_sync_policy.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "../gen_block_tensor/impl/block_map_impl.h"
#include "block_tensor_traits.h"

namespace libtensor {

/** \brief Block %tensor
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
    typedef block_tensor_traits<T, Alloc> bt_traits;

public:
    block_index_space<N> m_bis; //!< Block %index space
    dimensions<N> m_bidims; //!< Block %index %dimensions
    symmetry<N, T> m_symmetry; //!< Block %tensor symmetry
    orbit_list<N, T> *m_orblst; //!< Orbit list
    bool m_orblst_dirty; //!< Whether the orbit list needs to be updated
    block_map<N, T, bt_traits> m_map; //!< Block map
    block_map<N, T, bt_traits> m_aux_map; //!< Auxiliary block map
    rwlock_t *m_lock; //!< Read-write lock
    mutex_t *m_locku; //!< Upgrade lock

public:
    //!    \name Construction and destruction
    //@{
    block_tensor(const block_index_space<N> &bis);
    block_tensor(const block_tensor<N, T, Alloc> &bt);
    virtual ~block_tensor();
    //@}

    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{
    virtual const block_index_space<N> &get_bis() const;
    //@}

protected:
    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{
    virtual const symmetry<N, T> &on_req_const_symmetry();
    virtual symmetry<N, T> &on_req_symmetry();
    virtual dense_tensor_i<N, T> &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual dense_tensor_i<N, T> &on_req_aux_block(const index<N> &idx);
    virtual void on_ret_aux_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    virtual void on_req_sync_on();
    virtual void on_req_sync_off();
    //@}

    //!    \name Implementation of libtensor::immutable
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
    m_map(m_bis),
    m_aux_map(m_bis),
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
    m_map(m_bis),
    m_aux_map(m_bis),
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
const symmetry<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_const_symmetry() {

    return m_symmetry;
}


template<size_t N, typename T, typename Alloc, typename Sync>
symmetry<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_symmetry() {

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
dense_tensor_i<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_const_block(
    const index<N> &idx) {

    return on_req_block(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_ret_const_block(const index<N> &idx) {

    on_ret_block(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
dense_tensor_i<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_block(
    const index<N> &idx) {

    static const char *method = "on_req_block(const index<N>&)";

    auto_lock lock(m_lock, m_locku, false);

    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    if(!m_map.contains(idx)) {
        lock.upgrade();
        m_map.create(idx);
        lock.downgrade();
    }
    return m_map.get(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_ret_block(const index<N> &idx) {

}


template<size_t N, typename T, typename Alloc, typename Sync>
dense_tensor_i<N, T> &block_tensor<N, T, Alloc, Sync>::on_req_aux_block(
    const index<N> &idx) {

    static const char *method = "on_req_aux_block(const index<N>&)";

    auto_lock lock(m_lock, m_locku, true);

    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    if(m_aux_map.contains(idx)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Duplicate aux request.");
    }

    m_aux_map.create(idx);

    return m_aux_map.get(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_ret_aux_block(const index<N> &idx) {

    auto_lock lock(m_lock, m_locku, true);

    m_aux_map.remove(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
bool block_tensor<N, T, Alloc, Sync>::on_req_is_zero_block(
    const index<N> &idx) {

    static const char *method = "on_req_is_zero_block(const index<N>&)";

    auto_lock lock(m_lock, m_locku, false);

    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    return !m_map.contains(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_zero_block(const index<N> &idx) {

    static const char *method = "on_req_zero_block(const index<N>&)";

    auto_lock lock(m_lock, m_locku, true);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    m_map.remove(idx);
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_zero_all_blocks() {

    static const char *method = "on_req_zero_all_blocks()";

    auto_lock lock(m_lock, m_locku, true);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    m_map.clear();
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_sync_on() {

    if(m_lock == 0) {
        m_lock = new rwlock_t;
        m_locku = new mutex_t;
    }
}


template<size_t N, typename T, typename Alloc, typename Sync>
void block_tensor<N, T, Alloc, Sync>::on_req_sync_off() {

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
