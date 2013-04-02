#ifndef LIBTENSOR_CUDA_ALLOCATOR_H
#define LIBTENSOR_CUDA_ALLOCATOR_H

#include <cstddef> // for size_t
#include "cuda_pointer.h"

namespace libtensor {


/** \brief Simple allocator for CUDA GPU memory
    \tparam T Data type.

    This class allocates memory on nVidia CUDA GPUs using cudaMalloc/cudaFree.

    \ingroup libtensor_cuda
 **/
template<typename T>
class cuda_allocator {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef cuda_pointer<T> pointer_type;
//    typedef struct cuda_pointer {
//        T *p;
//        cuda_pointer(T *p_ = 0) : p(p_) { }
//        bool operator==(const cuda_pointer &p_) const { return p == p_.p; }
//    } pointer_type; //!< Wrapped CUDA pointer type

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

public:
    /** \brief Allocates a block of GPU memory
        \param sz Block size (in units of T).
        \return Pointer to the block of GPU memory.
        \throw out_of_memory If not enough memory is available.
        \throw cuda_exception In case of CUDA errors.
     **/
    static pointer_type allocate(size_t sz);

    /** \brief Deallocates (frees) a block of memory previously
            allocated using allocate()
        \param p Pointer to a block of GPU memory.
        \throw cuda_exception In case of CUDA errors.
     **/
    static void deallocate(pointer_type p);

    /** \brief Prefetches a block of memory (does nothing in this
            implementation)
        \param p Pointer to a block of GPU memory.
     **/
    static void prefetch(pointer_type p) {

    }

    /** \brief Locks a block of memory in physical space for read-only
            (does nothing in this implementation)
        \param p Pointer to a block of GPU memory.
        \return Constant physical pointer to the memory.
     **/
    static cuda_pointer<const T> lock_ro(pointer_type p) {

    	cuda_pointer<const T> tmp_p( p.get_physical_pointer() );
        return tmp_p;
    }

    /** \brief Unlocks a block of memory previously locked by lock_ro()
            (does nothing in this implementation)
        \param p Pointer to a block of GPU memory.
     **/
    static void unlock_ro(pointer_type p) {

    }

    /** \brief Locks a block of memory in physical space for read-write
            (does nothing in this implementation)
        \param p Pointer to a block of GPU memory.
        \return Physical pointer to the memory.
     **/
    static cuda_pointer<T> lock_rw(pointer_type p) {

    	return p;
    }

    /** \brief Unlocks a block of memory previously locked by lock_rw()
            (does nothing in this implementation)
        \param p Pointer to a block of GPU memory.
     **/
    static void unlock_rw(pointer_type p) {

    }

    static void set_priority(pointer_type p) {

    }

    static void unset_priority(pointer_type p) {

    }

    /** \brief Copies a block of memory from host memory to device memory
        \param dp Pointer to a block of GPU memory on the device (destination).
        \param hp Pointer to a block of memory on the host (source).
        \param sz Size of block in the units of T.
     **/
    static void copy_to_device(pointer_type dp, const T *hp, size_t sz);

    /** \brief Copies a block of memory from device memory to host memory
        \param hp Pointer to a block of memory on the host (destination).
        \param dp Pointer to a block of GPU memory on the device (source).
        \param sz Size of block in the units of T.
     **/
    static void copy_to_host(T *hp, pointer_type dp, size_t sz);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_ALLOCATOR_H
