#ifndef LIBTENSOR_ALLOCATOR_H
#define LIBTENSOR_ALLOCATOR_H

#include <cstdlib> // for size_t
#include <string>

namespace libtensor {


/** \brief Abstract base class for the wrapped allocator implementation
    \tparam T Data type

    The wrapper is used by the allocator to encapsulate currently activated
    allocator implementation.

    \ingroup libtensor_core
 **/
template<typename T>
class allocator_wrapper_i {
public:
    typedef struct pointer {
        private: T *x;
    } pointer_type;

public:
    virtual ~allocator_wrapper_i() { }

    virtual void init(const char *pfprefix) = 0;
    virtual void shutdown() = 0;
    virtual size_t get_block_size(size_t sz) = 0;
    virtual pointer_type allocate(size_t sz) = 0;
    virtual void deallocate(const pointer_type &p) noexcept = 0;
    virtual void prefetch(const pointer_type &p) = 0;
    virtual T *lock_rw(const pointer_type &p) = 0;
    virtual const T *lock_ro(const pointer_type &p) = 0;
    virtual void unlock_rw(const pointer_type &p) = 0;
    virtual void unlock_ro(const pointer_type &p) = 0;
    virtual void set_priority(const pointer_type &p) = 0;
    virtual void unset_priority(const pointer_type &p) = 0;

};


/** \brief Memory allocator used in the tensor library

    \ingroup libtensor_core
 **/
template<typename T>
class allocator {
public:
    typedef typename allocator_wrapper_i<T>::pointer_type pointer_type;

public:
    static const pointer_type invalid_pointer; //!< Invalid pointer constant

private:
    static allocator_wrapper_i<T> *m_aimpl; //!< Implementation

public:
    /** \brief Initializes the allocator with a given implementation
        \param pfprefix Prefix to page file path.
     **/
    static void init(const std::string &implementation, const char *pfprefix = 0);

    /** \brief Initializes the allocator with the default implementation
            (std_allocator)
        \param pfprefix Prefix to page file path.
     **/
    static void init() { init("standard", NULL); }

    /** Old init function for compatibility. Deprecated */
    static void init(const std::string &implementation, size_t, size_t, size_t,
                     size_t, const char *pfprefix = 0) {
        init(implementation, pfprefix);
    }

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
    static void deallocate(const pointer_type &p) {
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
    static T *lock_rw(const pointer_type &p) {
        return m_aimpl->lock_rw(p);
    }

    /** \brief Makes a block available and locks it in physical memory
            for reading only
        \param p Virtual memory pointer.
        \return Constant physical pointer.
     **/
    static const T *lock_ro(const pointer_type &p) {
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
    static allocator_wrapper_i<T> *make_default_allocator();

};


} // namespace libtensor

#endif // LIBTENSOR_ALLOCATOR_H
