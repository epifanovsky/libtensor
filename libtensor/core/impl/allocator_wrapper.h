#ifndef LIBTENSOR_ALLOCATOR_WRAPPER_H
#define LIBTENSOR_ALLOCATOR_WRAPPER_H

#include "../allocator.h"

namespace libtensor {


/** \brief Wrapper class for an allocator implementation
    \tparam T Data type
    \tparam AllocatorImpl Implementation class

    \ingroup libtensor_core
 **/
template<typename AllocatorImpl>
class allocator_wrapper : public allocator_wrapper_i {
public:
    typedef typename allocator_wrapper_i::pointer_type internal_pointer_type;
    typedef typename AllocatorImpl::pointer_type external_pointer_type;
    typedef internal_pointer_type pointer_type;

private:
    AllocatorImpl m_impl;

public:
    allocator_wrapper(const AllocatorImpl &impl = AllocatorImpl()) :
        m_impl(impl)
    { }

    virtual ~allocator_wrapper() { }

    virtual void init(size_t base_sz, size_t min_sz, size_t max_sz,
        size_t mem_limit, const char *pfprefix) {
        m_impl.init(base_sz, min_sz, max_sz, mem_limit, pfprefix);
    }

    virtual void shutdown() {
        m_impl.shutdown();
    }

    virtual size_t get_block_size(size_t sz) {
        return m_impl.get_block_size(sz);
    }

    virtual pointer_type allocate(size_t sz) {
        external_pointer_type p2 = m_impl.allocate(sz);
        return reinterpret_cast<const internal_pointer_type&>(p2);
    }

    virtual void deallocate(const pointer_type &p) throw () {
        m_impl.deallocate(convp(p));
    }

    virtual void prefetch(const pointer_type &p) {
        m_impl.prefetch(convp(p));
    }

    virtual void *lock_rw(const pointer_type &p) {
        return m_impl.lock_rw(convp(p));
    }

    virtual const void *lock_ro(const pointer_type &p) {
        return m_impl.lock_ro(convp(p));
    }

    virtual void unlock_rw(const pointer_type &p) {
        m_impl.unlock_rw(convp(p));
    }

    virtual void unlock_ro(const pointer_type &p) {
        m_impl.unlock_ro(convp(p));
    }

    virtual void set_priority(const pointer_type &p) {
        m_impl.set_priority(convp(p));
    }

    virtual void unset_priority(const pointer_type &p) {
        m_impl.unset_priority(convp(p));
    }

public:
    static pointer_type make_invalid_pointer() {
        external_pointer_type p2 = AllocatorImpl::invalid_pointer;
        return reinterpret_cast<const internal_pointer_type&>(p2);
    }

private:
    static external_pointer_type convp(const internal_pointer_type &p1) {
        const external_pointer_type &p2 =
            reinterpret_cast<const external_pointer_type&>(p1);
        return p2;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_WRAPPER_H

