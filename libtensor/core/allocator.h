#ifndef LIBTENSOR_ALLOCATOR_H
#define LIBTENSOR_ALLOCATOR_H

#include <libvmm/ec_allocator.h>
#include <libvmm/std_allocator.h>
#include <libvmm/vm_allocator.h>
#include <libvmm/evmm/evmm.h>

namespace libtensor {


/** \brief Virtual memory allocator used in the tensor library

    This memory allocator uses libvmm::vm_allocator.

    \sa ec_allocator_base

    \ingroup libtensor_core
 **/
template<typename T>
class vm_allocator_base : public libvmm::vm_allocator< T, libvmm::evmm<T> > {
public:
    typedef libvmm::evmm<T> vmm_type;
    typedef libvmm::vm_allocator<T, vmm_type> vm_allocator_type;

public:
    /** \brief Returns the reference to the virtual memory manager
     **/
    static vmm_type &vmm() {
        return vm_allocator_type::vmm();
    }

};


/** \brief Error-checking memory allocator used in the tensor library

    This memory allocator uses libvmm::ec_allocator.

    \sa vm_allocator_base

    \ingroup libtensor_core
 **/
template<typename T>
class ec_allocator_base :
    public libvmm::ec_allocator< T, libvmm::vm_allocator< T, libvmm::evmm<T> >,
        libvmm::std_allocator<T> > {

public:
    typedef libvmm::evmm<T> vmm_type;
    typedef libvmm::vm_allocator<T, vmm_type> vm_allocator_type;

public:
    /** \brief Returns the reference to the virtual memory manager
     **/
    static vmm_type &vmm() {
        return vm_allocator_type::vmm();
    }

};


/** \brief Memory allocator used in the tensor library

    This is a proxy class that uses the error-checking allocator if
    LIBTENSOR_DEBUG is defined or the regular virtual memory allocator
    otherwise.

    \sa ec_allocator_base, vm_allocator_base

    \ingroup libtensor_core
 **/
template<typename T>
class allocator :
#ifdef LIBTENSOR_DEBUG
    public ec_allocator_base<T> {
#else
    public vm_allocator_base<T> {
#endif // LIBTENSOR_DEBUG

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "allocator_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_ALLOCATOR_H
