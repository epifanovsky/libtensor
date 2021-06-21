#ifndef LIBTENSOR_ALLOCATOR_WRAPPER_H
#define LIBTENSOR_ALLOCATOR_WRAPPER_H

#include "../allocator.h"

namespace libtensor {


/** \brief Wrapper class for an allocator implementation
    \tparam T Data type
    \tparam AllocatorImpl Implementation class

    \ingroup libtensor_core
 **/
template<typename T, typename AllocatorImpl>
class allocator_wrapper : public allocator_wrapper_i<T> {
public:
    typedef typename allocator_wrapper_i<T>::pointer_type pointer_type;

private:
    union pointer {
        pointer_type p1;
        typename AllocatorImpl::pointer_type p2;
    };

private:
    AllocatorImpl m_impl;

public:
    allocator_wrapper(const AllocatorImpl &impl = AllocatorImpl()) :
        m_impl(impl)
    { }

    virtual ~allocator_wrapper() { }

    virtual void init(const char *pfprefix) {
        m_impl.init(pfprefix);
    }

    virtual void shutdown() {
        m_impl.shutdown();
    }

    virtual size_t get_block_size(size_t sz) {
        return m_impl.get_block_size(sz);
    }

    virtual pointer_type allocate(size_t sz) {
        pointer ptr;
        ptr.p2 = m_impl.allocate(sz);
        return ptr.p1;
    }

    virtual void deallocate(const pointer_type &p) noexcept {
        m_impl.deallocate(convp(p));
    }

    virtual void prefetch(const pointer_type &p) {
        m_impl.prefetch(convp(p));
    }

    virtual T *lock_rw(const pointer_type &p) {
        return m_impl.lock_rw(convp(p));
    }

    virtual const T *lock_ro(const pointer_type &p) {
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
        pointer ptr;
        ptr.p2 = AllocatorImpl::invalid_pointer;
        return ptr.p1;
    }

private:
    static typename AllocatorImpl::pointer_type convp(const pointer_type &p) {
        pointer ptr;
        ptr.p1 = p;
        return ptr.p2;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_WRAPPER_H

