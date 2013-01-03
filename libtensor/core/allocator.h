#ifndef LIBTENSOR_ALLOCATOR_H
#define LIBTENSOR_ALLOCATOR_H

#ifndef WITHOUT_LIBVMM

#ifdef LIBTENSOR_ALLOCATOR_DEBUG
#include <libvmm/ec_allocator.h>
#endif // LIBTENSOR_ALLOCATOR_DEBUG

#include <libvmm/vm_allocator.h>
#include <libvmm/evmm/evmm.h>

#endif // WITHOUT_LIBVMM

#include "std_allocator.h"

namespace libtensor {


#ifndef WITHOUT_LIBVMM

/** \brief Virtual memory allocator used in the tensor library

    This memory allocator uses libvmm::vm_allocator or std_allocator if
    libvmm is unavailable.

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


#ifdef LIBTENSOR_ALLOCATOR_DEBUG

/** \brief Error-checking memory allocator used in the tensor library

    This memory allocator uses libvmm::ec_allocator.

    \sa vm_allocator_base

    \ingroup libtensor_core
 **/
template<typename T>
class ec_allocator_base :
    public libvmm::ec_allocator< T, libvmm::vm_allocator< T, libvmm::evmm<T> >,
        std_allocator<T> > {

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

#endif // LIBTENSOR_ALLOCATOR_DEBUG

#endif // WITHOUT_LIBVMM


/** \brief Memory allocator used in the tensor library

    This is a proxy class that uses the error-checking allocator if
    LIBTENSOR_ALLOCATOR_DEBUG is defined or the regular virtual memory
    allocator otherwise.

    \sa ec_allocator_base, vm_allocator_base, std_allocator

    \ingroup libtensor_core
 **/
template<typename T>
class allocator :
#ifdef WITHOUT_LIBVMM
    public std_allocator<T> {
#else // WITHOUT_LIBVMM
#ifdef LIBTENSOR_ALLOCATOR_DEBUG
    public ec_allocator_base<T> {
#else // LIBTENSOR_ALLOCATOR_DEBUG
    public vm_allocator_base<T> {
#endif // LIBTENSOR_ALLOCATOR_DEBUG
#endif // WITHOUT_LIBVMM

};


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_H
