#ifndef LIBTENSOR_VM_ALLOCATOR_H
#define LIBTENSOR_VM_ALLOCATOR_H

#include <libvmm/evmm/evmm.h>

namespace libtensor {


/** \brief Virtual memory allocator
    \tparam T Data type.

    \ingroup libtensor_core
 **/
template<typename T>
class vm_allocator {
public:
    typedef libvmm::evmm<char> vmm;
    typedef typename vmm::vm_pointer_type pointer_type; //!< Pointer type

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

public:
    static size_t m_base_sz;
    static size_t m_min_sz;
    static size_t m_max_sz;

public:
    /** \brief Initializes the virtual memory manager

        \param base_sz Exponential base for block size increment.
        \param min_sz Smallest block size in bytes.
        \param max_sz Largest block size in bytes.
        \param mem_limit Memory limit in bytes.
     **/
    static void init(size_t base_sz, size_t min_sz, size_t max_sz,
        size_t mem_limit, const char *prefix = 0) {

        if(prefix == 0) {
            typename vmm::page_file_factory_type pff;
            vmm::get_instance().init(base_sz, min_sz, max_sz, mem_limit, pff);
        } else {
            typename vmm::page_file_factory_type pff(prefix);
            vmm::get_instance().init(base_sz, min_sz, max_sz, mem_limit, pff);
        }

        m_base_sz = base_sz;
        m_min_sz = min_sz;
        m_max_sz = max_sz;
    }

    /** \brief Shuts down the memory manager

        The virtual memory manager is not usable after it is shut down until
        it is initialized again via init().

        This method frees all the memory allocated by the memory manager.
     **/
    static void shutdown() {
        vmm::get_instance().shutdown();
    }

    /** \brief Returns the real size of a block, in bytes, including alignment
        \param sz Block size in units of T.
     **/
    static size_t get_block_size(size_t sz) {
        size_t real_sz = m_min_sz;
        sz *= sizeof(T);
        if(sz > m_max_sz) {
            real_sz = sz;
        } else {
            while(real_sz < sz) real_sz *= m_base_sz;
        }
        return real_sz;
    }

    /** \brief Allocates a block of memory
        \param sz Block size (in units of type T).
        \return Pointer to the block of memory.
     **/
    static pointer_type allocate(size_t sz) {
        return vmm::get_instance().allocate(sz * sizeof(T));
    }

    /** \brief Deallocates (frees) a block of memory previously
            allocated using allocate()
        \param p Pointer to the block of memory.
     **/
    static void deallocate(pointer_type p) {
        vmm::get_instance().deallocate(p);
    }

    /** \brief Prefetches a block of memory (does nothing in this
            implementation)
        \param p Pointer to the block of memory.
     **/
    static void prefetch(pointer_type p) {
        vmm::get_instance().prefetch(p);
    }

    /** \brief Locks a block of memory in physical space for read-only
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
        \return Constant physical pointer to the memory.
     **/
    static const T *lock_ro(pointer_type p) {
        return reinterpret_cast<const T*>(vmm::get_instance().lock_ro(p));
    }

    /** \brief Unlocks a block of memory previously locked by lock_ro()
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
     **/
    static void unlock_ro(pointer_type p) {
        vmm::get_instance().unlock_ro(p);
    }

    /** \brief Locks a block of memory in physical space for read-write
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
        \return Physical pointer to the memory.
     **/
    static T *lock_rw(pointer_type p) {
        return reinterpret_cast<T*>(vmm::get_instance().lock_rw(p));
    }

    /** \brief Unlocks a block of memory previously locked by lock_rw()
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
     **/
    static void unlock_rw(pointer_type p) {
        vmm::get_instance().unlock_rw(p);
    }

    /** \brief Sets a priority flag on a memory block (stub)
        \param p Pointer to a block of memory.
     **/
    static void set_priority(pointer_type p) {
        vmm::get_instance().set_priority(p);
    }

    /** \brief Unsets a priority flag on a memory block (stub)
        \param p Pointer to a block of memory.
     **/
    static void unset_priority(pointer_type p) {
        vmm::get_instance().unset_priority(p);
    }

};


template<typename T>
const typename vm_allocator<T>::pointer_type
    vm_allocator<T>::invalid_pointer = libvmm::evmm<char>::invalid_pointer;

template<typename T> size_t vm_allocator<T>::m_base_sz;
template<typename T> size_t vm_allocator<T>::m_min_sz;
template<typename T> size_t vm_allocator<T>::m_max_sz;


} // namespace libtensor

#endif // LIBTENSOR_VM_ALLOCATOR_H
