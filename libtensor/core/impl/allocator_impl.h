#ifndef LIBTENSOR_ALLOCATOR_IMPL_H
#define LIBTENSOR_ALLOCATOR_IMPL_H

#if !defined(WITHOUT_LIBVMM)
#ifdef LIBTENSOR_ALLOCATOR_DEBUG
#include <libvmm/ec_allocator.h>
#endif // LIBTENSOR_ALLOCATOR_DEBUG
#include <libvmm/vm_allocator.h>
#include <libvmm/evmm/evmm.h>
#endif // WITHOUT_LIBVMM

#include "../allocator.h"

namespace libtensor {

#if !defined(WITHOUT_LIBVMM)

#ifdef LIBTENSOR_ALLOCATOR_DEBUG

template<typename T>
class allocator_base :
    public libvmm::ec_allocator< T, libvmm::vm_allocator< T, libvmm::evmm<T> >,
        std_allocator<T> > {

public:
    typedef libvmm::evmm<T> vmm_type;
    typedef libvmm::vm_allocator<T, vmm_type> vm_allocator_type;

};

#else // LIBTENSOR_ALLOCATOR_DEBUG

template<typename T>
class allocator_base : public libvmm::vm_allocator< T, libvmm::evmm<T> > {
public:
    typedef libvmm::evmm<T> vmm_type;
    typedef libvmm::vm_allocator<T, vmm_type> vm_allocator_type;

};

#endif // LIBTENSOR_ALLOCATOR_DEBUG


template<typename T>
const typename allocator<T>::pointer_type allocator<T>::invalid_pointer =
    allocator_base<T>::vmm_type::invalid_pointer;

template<typename T>
size_t allocator<T>::m_base_sz = 0;


template<typename T>
size_t allocator<T>::m_min_sz = 0;


template<typename T>
size_t allocator<T>::m_max_sz = 0;


template<typename T>
void allocator<T>::init(size_t base_sz, size_t min_sz, size_t max_sz,
    size_t mem_limit, const char *pfprefix) {

    if(pfprefix == 0) {
        typename allocator_base<T>::vmm_type::page_file_factory_type pff;
        allocator_base<T>::vmm_type::get_instance().
            init(base_sz, min_sz, max_sz, mem_limit, pff);
    } else {
        typename allocator_base<T>::vmm_type::page_file_factory_type
            pff(pfprefix);
        allocator_base<T>::vmm_type::get_instance().
            init(base_sz, min_sz, max_sz, mem_limit, pff);
    }
    m_base_sz = base_sz;
    m_min_sz = min_sz;
    m_max_sz = max_sz;
}


template<typename T>
void allocator<T>::shutdown() {

    allocator_base<T>::vmm_type::get_instance().shutdown();
}


template<typename T>
size_t allocator<T>::get_block_size(size_t sz) {

    size_t real_sz = m_min_sz;
    if(sz > m_max_sz) {
        real_sz = sz;
    } else {
        while(real_sz < sz) real_sz *= m_base_sz;
    }
    return real_sz * sizeof(T);
}


template<typename T>
typename allocator<T>::pointer_type allocator<T>::allocate(size_t sz) {

    return allocator_base<T>::vmm_type::get_instance().allocate(sz);
}


template<typename T>
void allocator<T>::deallocate(const pointer_type &p) throw () {

    allocator_base<T>::vmm_type::get_instance().deallocate(p);
}


template<typename T>
void allocator<T>::prefetch(const pointer_type &p) {

    allocator_base<T>::vmm_type::get_instance().prefetch(p);
}


template<typename T>
T *allocator<T>::lock_rw(const pointer_type &p) {

    return allocator_base<T>::vmm_type::get_instance().lock_rw(p);
}


template<typename T>
const T *allocator<T>::lock_ro(const pointer_type &p) {

    return allocator_base<T>::vmm_type::get_instance().lock_ro(p);
}


template<typename T>
void allocator<T>::unlock_rw(const pointer_type &p) {

    allocator_base<T>::vmm_type::get_instance().unlock_rw(p);
}


template<typename T>
void allocator<T>::unlock_ro(const pointer_type &p) {

    allocator_base<T>::vmm_type::get_instance().unlock_ro(p);
}


template<typename T>
void allocator<T>::set_priority(const pointer_type &p) {

    allocator_base<T>::vmm_type::get_instance().set_priority(p);
}


template<typename T>
void allocator<T>::unset_priority(const pointer_type &p) {

    allocator_base<T>::vmm_type::get_instance().unset_priority(p);
}


#endif // WITHOUT_LIBVMM

} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_IMPL_H
