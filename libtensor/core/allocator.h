#ifndef LIBTENSOR_ALLOCATOR_H
#define LIBTENSOR_ALLOCATOR_H

#if !defined(WITHOUT_LIBVMM)
#include <libvmm/vm_handle.h>
#endif // WITHOUT_LIBVMM

#include "std_allocator.h"

namespace libtensor {


#if !defined(WITHOUT_LIBVMM)

/** \brief Memory allocator used in the tensor library

    This memory allocator uses libvmm::vm_allocator (default),
    libvmm::ec_allocator (if LIBTENSOR_ALLOCATOR_DEBUG is defined),
    or std_allocator if libvmm is unavailable.

    \ingroup libtensor_core
 **/
template<typename T>
class allocator {
public:
    typedef libvmm::vm_handle pointer_type;

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

public:
    /** \brief Initializes the virtual memory manager

        \param base_sz Exponential base for block size increment.
        \param min_sz Smallest block size in data elements.
        \param max_sz Largest block size in data elements.
        \param mem_limit Memory limit in data elements.
     **/
    static void init(size_t base_sz, size_t min_sz, size_t max_sz, size_t mem_limit);

    /** \brief Shuts down the virtual memory manager

        The virtual memory manager is not usable after it is shut down until
        it is initialized again via init().

        This method frees all the memory allocated by the memory manager.
     **/
    static void shutdown();

    /** \brief Allocates a block of memory
        \param sz Block size in units of T.
        \return Virtual memory pointer.
     **/
    static pointer_type allocate(size_t sz);

    /** \brief Deallocates (frees) a block of memory previously
            allocated by allocate()
        \param p Virtual memory pointer.
     **/
    static void deallocate(const pointer_type &p) throw ();

    /** \brief Informs the virtual memory manager that a block of
            memory may soon be locked
        \param p Virtual memory pointer.
     **/
    static void prefetch(const pointer_type &p);

    /** \brief Makes a block available and locks it in physical memory
            for reading and writing
        \param p Virtual memory pointer.
        \return Physical pointer.
     **/
    static T *lock_rw(const pointer_type &p);

    /** \brief Makes a block available and locks it in physical memory
            for reading only
        \param p Virtual memory pointer.
        \return Constant physical pointer.
     **/
    static const T *lock_ro(const pointer_type &p);

    /** \brief Unlocks a block previously locked by lock_rw()
        \param p Virtual memory pointer.
     **/
    static void unlock_rw(const pointer_type &p);

    /** \brief Unlocks a block previously locked by lock_ro()
        \param p Virtual memory pointer.
     **/
    static void unlock_ro(const pointer_type &p);

    /** \brief Sets a priority flag on a virtual memory block
        \param p Virtual memory pointer.
     **/
    static void set_priority(const pointer_type &p);

    /** \brief Unsets a priority flag on a virtual memory block
        \param p Virtual memory pointer.
     **/
    static void unset_priority(const pointer_type &p);

};

#else // WITHOUT_LIBVMM

template<typename T>
class allocator : public std_allocator<T> { };

#endif // WITHOUT_LIBVMM


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_H
