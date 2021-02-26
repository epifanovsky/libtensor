#ifndef LIBTENSOR_ALLOCATOR_H
#define LIBTENSOR_ALLOCATOR_H

#include <cstdlib> // for size_t

namespace libtensor {


/** \brief Abstract base class for the wrapped allocator implementation
    \tparam T Data type

    The wrapper is used by the allocator to encapsulate currently activated
    allocator implementation.

    \ingroup libtensor_core
 **/
//template<typename T>
class allocator_wrapper_i {
public:
    typedef struct pointer {
        //private: T *x;
        private: void *x;
    } pointer_type;

public:
    virtual ~allocator_wrapper_i() { }

    virtual void init(size_t base_sz, size_t min_sz, size_t max_sz,
        size_t mem_limit, const char *pfprefix) = 0;
    virtual void shutdown() = 0;
    virtual size_t get_block_size(size_t sz) = 0;
    virtual pointer_type allocate(size_t sz) = 0;
    virtual void deallocate(const pointer_type &p) throw () = 0;
    virtual void prefetch(const pointer_type &p) = 0;
    //virtual T *lock_rw(const pointer_type &p) = 0;
    virtual void *lock_rw(const pointer_type &p) = 0;
    //virtual const T *lock_ro(const pointer_type &p) = 0;
    virtual const void *lock_ro(const pointer_type &p) = 0;
    virtual void unlock_rw(const pointer_type &p) = 0;
    virtual void unlock_ro(const pointer_type &p) = 0;
    virtual void set_priority(const pointer_type &p) = 0;
    virtual void unset_priority(const pointer_type &p) = 0;

};


/** \brief Memory allocator used in the tensor library

    \ingroup libtensor_core
 **/
class allocator {
public:
    typedef allocator_wrapper_i::pointer_type pointer_type;

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

private:
    static allocator_wrapper_i *m_aimpl; //!< Implementation
    static size_t m_base_sz; //!< Exponential base of data block size
    static size_t m_min_sz; //!< Smallest block size in data elements
    static size_t m_max_sz; //!< Largest block size in data elements

public:
    /** \brief Initializes the allocator with a given implementation

        \param base_sz Exponential base of data block size.
        \param min_sz Smallest block size in data elements.
        \param max_sz Largest block size in data elements.
        \param mem_limit Memory limit in data elements.
        \param pfprefix Prefix to page file path.
     **/
    template<typename AllocatorImpl>
    static void init(const AllocatorImpl &aimpl, size_t base_sz, size_t min_sz,
        size_t max_sz, size_t mem_limit, const char *pfprefix = 0);

    /** \brief Initializes the allocator with the default implementation
            (std_allocator)

        \param base_sz Exponential base of data block size.
        \param min_sz Smallest block size in data elements.
        \param max_sz Largest block size in data elements.
        \param mem_limit Memory limit in data elements.
        \param pfprefix Prefix to page file path.
     **/
    static void init(size_t base_sz, size_t min_sz, size_t max_sz,
        size_t mem_limit, const char *pfprefix = 0);

    /** \brief Shuts down the allocator

        The virtual memory manager is not usable after it is shut down until
        it is initialized again via init().

        This method frees all the memory allocated by the memory manager.
     **/
    static void shutdown();

    /** \brief Returns the real size of a block, in bytes, including alignment
        \param sz Block size in units of T.
     **/
    static size_t get_block_size(size_t sz) {
        return m_aimpl->get_block_size(sz);
    }

    /** \brief Allocates a block of memory
        \param sz Block size in units of T.
        \return Virtual memory pointer.
     **/
    static pointer_type allocate(size_t sz) {
        return m_aimpl->allocate(sz);
    }

    /** \brief Deallocates (frees) a block of memory previously
            allocated by allocate()
        \param p Virtual memory pointer.
     **/
    static void deallocate(const pointer_type &p) throw () {
        m_aimpl->deallocate(p);
    }

    /** \brief Informs the virtual memory manager that a block of
            memory may soon be locked
        \param p Virtual memory pointer.
     **/
    static void prefetch(const pointer_type &p) {
        m_aimpl->prefetch(p);
    }

    /** \brief Makes a block available and locks it in physical memory
            for reading and writing
        \param p Virtual memory pointer.
        \return Physical pointer.
     **/
    static void *lock_rw(const pointer_type &p) {
        return m_aimpl->lock_rw(p);
    }

    /** \brief Makes a block available and locks it in physical memory
            for reading only
        \param p Virtual memory pointer.
        \return Constant physical pointer.
     **/
    static const void *lock_ro(const pointer_type &p) {
        return m_aimpl->lock_ro(p);
    }

    /** \brief Unlocks a block previously locked by lock_rw()
        \param p Virtual memory pointer.
     **/
    static void unlock_rw(const pointer_type &p) {
        m_aimpl->unlock_rw(p);
    }

    /** \brief Unlocks a block previously locked by lock_ro()
        \param p Virtual memory pointer.
     **/
    static void unlock_ro(const pointer_type &p) {
        m_aimpl->unlock_ro(p);
    }

    /** \brief Sets a priority flag on a virtual memory block
        \param p Virtual memory pointer.
     **/
    static void set_priority(const pointer_type &p) {
        m_aimpl->set_priority(p);
    }

    /** \brief Unsets a priority flag on a virtual memory block
        \param p Virtual memory pointer.
     **/
    static void unset_priority(const pointer_type &p) {
        m_aimpl->unset_priority(p);
    }

private:
    static pointer_type make_invalid_pointer();
    static allocator_wrapper_i *make_default_allocator();

};


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_H
