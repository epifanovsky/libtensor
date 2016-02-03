#ifndef LIBTENSOR_ALLOCATOR_IMPL_H
#define LIBTENSOR_ALLOCATOR_IMPL_H

#include "allocator_wrapper.h"
#include "std_allocator.h"
#include "../allocator.h"
#include "../allocator_init.h"

#if !defined(WITHOUT_LIBVMM)
#include "vm_allocator.h"
#endif

#if defined(USE_LIBXM)
#include "xm_allocator.h"
#endif

namespace libtensor {


template<typename T>
struct default_allocator {
    typedef std_allocator<T> type;
};


template<typename T>
const typename allocator<T>::pointer_type allocator<T>::invalid_pointer(
    allocator<T>::make_invalid_pointer());

template<typename T>
allocator_wrapper_i<T> *allocator<T>::m_aimpl(
    allocator<T>::make_default_allocator());

template<typename T> size_t allocator<T>::m_base_sz = 0;
template<typename T> size_t allocator<T>::m_min_sz = 0;
template<typename T> size_t allocator<T>::m_max_sz = 0;


template<typename T>
typename allocator<T>::pointer_type allocator<T>::make_invalid_pointer() {

    typedef typename default_allocator<T>::type defalloc;
    return allocator_wrapper<T, defalloc>::make_invalid_pointer();
}

template<typename T>
allocator_wrapper_i<T> *allocator<T>::make_default_allocator() {

    typedef typename default_allocator<T>::type defalloc;
    static allocator_wrapper<T, defalloc> a;
    return &a;
}


template<typename T>
void allocator<T>::init(size_t base_sz, size_t min_sz, size_t max_sz,
    size_t mem_limit, const char *pfprefix) {

#if defined(WITHOUT_LIBVMM)
#if defined(USE_LIBXM)
    typedef lt_xm_allocator::lt_xm_allocator<T> allocator_impl;
#else
    typedef std_allocator<T> allocator_impl;
#endif
#else
    typedef vm_allocator<T> allocator_impl;
#endif

    init(allocator_impl(), base_sz, min_sz, max_sz, mem_limit, pfprefix);
}


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_IMPL_H
