#ifndef LIBTENSOR_STD_ALLOCATOR_H
#define LIBTENSOR_STD_ALLOCATOR_H

#include <new>

namespace libtensor {


/** \brief Simple allocator based on C++ new and delete
    \tparam T Data type.

    The allocator is the interface to the virtual memory state machine.
    This simple implementation provides methods that are based on C++
    new and delete operators. Because there is no virtual memory involved
    here, the virtual and physical pointers are identical.

    See method descriptions below for more information.

    \ingroup libtensor_core
 **/
template<typename T>
class std_allocator {
public:
    typedef T *pointer_type; //!< Pointer type

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

public:
    /** \brief Initializes the virtual memory manager

        \param base_sz Exponential base for block size increment.
        \param min_sz Smallest block size in data elements.
        \param max_sz Largest block size in data elements.
        \param mem_limit Memory limit in data elements.
     **/
    static void init(size_t base_sz, size_t min_sz, size_t max_sz, size_t mem_limit) {

    }

    /** \brief Shuts down the memory manager

        The virtual memory manager is not usable after it is shut down until
        it is initialized again via init().

        This method frees all the memory allocated by the memory manager.
     **/
    static void shutdown() {

    }

    /** \brief Returns the real size of a block, in bytes, including alignment
        \param sz Block size in units of T.
     **/
    static size_t get_block_size(size_t sz) {
        return sz * sizeof(T)
    }

    /** \brief Allocates a block of memory
        \param sz Block size (in units of type T).
        \return Pointer to the block of memory.
     **/
    static pointer_type allocate(size_t sz) {
        return new T[sz];
    }

    /** \brief Deallocates (frees) a block of memory previously
            allocated using allocate()
        \param p Pointer to the block of memory.
     **/
    static void deallocate(pointer_type p) {
        delete [] p;
    }

    /** \brief Prefetches a block of memory (does nothing in this
            implementation)
        \param p Pointer to the block of memory.
     **/
    static void prefetch(pointer_type p) {

    }

    /** \brief Locks a block of memory in physical space for read-only
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
        \return Constant physical pointer to the memory.
     **/
    static const T *lock_ro(pointer_type p) {
        return p;
    }

    /** \brief Unlocks a block of memory previously locked by lock_ro()
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
     **/
    static void unlock_ro(pointer_type p) {

    }

    /** \brief Locks a block of memory in physical space for read-write
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
        \return Physical pointer to the memory.
     **/
    static T *lock_rw(pointer_type p) {
        return p;
    }

    /** \brief Unlocks a block of memory previously locked by lock_rw()
            (does nothing in this implementation)
        \param p Pointer to the block of memory.
     **/
    static void unlock_rw(pointer_type p) {

    }

    /** \brief Sets a priority flag on a memory block (stub)
        \param p Pointer to a block of memory.
     **/
    static void set_priority(pointer_type p) {

    }

    /** \brief Unsets a priority flag on a memory block (stub)
        \param p Pointer to a block of memory.
     **/
    static void unset_priority(pointer_type p) {

    }

};


template<typename T>
const typename std_allocator<T>::pointer_type
    std_allocator<T>::invalid_pointer = 0;


} // namespace libtensor

#endif // LIBTENSOR_STD_ALLOCATOR_H
