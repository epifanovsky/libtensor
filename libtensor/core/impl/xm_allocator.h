#ifndef LIBTENSOR_XM_ALLOCATOR_H
#define LIBTENSOR_XM_ALLOCATOR_H

#include <libtensor/core/batching_policy_base.h>
#include <libtensor/defs.h>
#include <libtensor/libxm/src/alloc.h>
#include <libutil/singleton.h>
#include <libutil/threads/auto_lock.h>
#include <libutil/threads/mutex.h>
#include <map>
#include <new>
#include <stdexcept>
#include <string>

namespace libtensor {
namespace lt_xm_allocator {

struct map_data {
  size_t size_bytes;
  int n_locks;
  void* lock_ptr;
};
typedef std::map<uintptr_t, map_data> map_type;

class alloc_data : public libutil::singleton<alloc_data> {
  friend class libutil::singleton<alloc_data>;

 public:
  struct xm_allocator* xm_allocator_inst;
  libutil::mutex g_lock;
  map_type g_map;
};

template <typename T>
class lt_xm_allocator {
 public:
  typedef uintptr_t pointer_type;  //!< Pointer type

 public:
  static const pointer_type invalid_pointer;  //!< Invalid pointer constant

 public:
  /** \brief Initializes the virtual memory manager

      \param base_sz Exponential base for block size increment.
      \param min_sz Smallest block size in data elements.
      \param max_sz Largest block size in data elements.
      \param mem_limit Memory limit in data elements.
   **/
  static void init(size_t base_sz, size_t min_sz, size_t max_sz, size_t mem_limit,
                   const char* prefix = 0) {

    std::string path;

    if (prefix) {
      std::string pref = prefix;
      path             = pref + "/" + "xmpagefile";
    }

    if (alloc_data::get_instance().xm_allocator_inst == NULL) {
      alloc_data::get_instance().xm_allocator_inst =
            xm_allocator_create(prefix ? path.c_str() : NULL);
    }
  }

  /** \brief Shuts down the memory manager

      The virtual memory manager is not usable after it is shut down until
      it is initialized again via init().

      This method frees all the memory allocated by the memory manager.
   **/
  static void shutdown() {
    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);

    if (alloc_data::get_instance().xm_allocator_inst) {
      xm_allocator_destroy(alloc_data::get_instance().xm_allocator_inst);
      alloc_data::get_instance().xm_allocator_inst = NULL;
      alloc_data::get_instance().g_map.clear();
    }
  }

  /** \brief Returns the real size of a block, in bytes, including alignment
      \param sz Block size in units of T.
   **/
  static size_t get_block_size(size_t sz) { return sz * sizeof(T); }

  /** \brief Allocates a block of memory
      \param sz Block size (in units of type T).
      \return Pointer to the block of memory.
   **/
  static pointer_type allocate(size_t sz) {
    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);
    map_data data;
    data.size_bytes = get_block_size(sz);
    data.lock_ptr   = NULL;
    data.n_locks    = 0;
    pointer_type p  = xm_allocator_allocate(alloc_data::get_instance().xm_allocator_inst,
                                           data.size_bytes);
    if (p == XM_NULL_PTR) throw std::runtime_error("allocate: unable to allocate memory");
    map_type::iterator it = alloc_data::get_instance().g_map.find(p);
    if (it != alloc_data::get_instance().g_map.end())
      throw std::runtime_error("allocate: pointer already allocated");
    alloc_data::get_instance().g_map.insert(std::pair<pointer_type, map_data>(p, data));
    return p;
  }

  /** \brief Deallocates (frees) a block of memory previously
          allocated using allocate()
      \param p Pointer to the block of memory.
   **/
  static void deallocate(pointer_type p) {
    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);
    map_type::iterator it = alloc_data::get_instance().g_map.find(p);
    if (it == alloc_data::get_instance().g_map.end())
      throw std::runtime_error("deallocate: pointer not allocated");
    if (it->second.n_locks != 0)
      throw std::runtime_error("deallocate: block still locked");
    alloc_data::get_instance().g_map.erase(it);
    xm_allocator_deallocate(alloc_data::get_instance().xm_allocator_inst, p);
  }

  /** \brief Prefetches a block of memory (does nothing in this
          implementation)
      \param p Pointer to the block of memory.
   **/
  static void prefetch(pointer_type p) {}

  /** \brief Locks a block of memory in physical space for read-only
          (does nothing in this implementation)
      \param p Pointer to the block of memory.
      \return Constant physical pointer to the memory.
   **/
  static const T* lock_ro(pointer_type p) {
    if (xm_allocator_get_path(alloc_data::get_instance().xm_allocator_inst) == NULL)
      return (const T*)p;

    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);
    map_type::iterator it = alloc_data::get_instance().g_map.find(p);
    if (it == alloc_data::get_instance().g_map.end())
      throw std::runtime_error("lock_ro: pointer not allocated");
    if (it->second.lock_ptr == NULL) {
      it->second.lock_ptr = malloc(it->second.size_bytes);
      if (it->second.lock_ptr == NULL) throw std::runtime_error("lock_ro: out of memory");
      xm_allocator_read(alloc_data::get_instance().xm_allocator_inst, p,
                        it->second.lock_ptr, it->second.size_bytes);
    }
    it->second.n_locks++;
    return (const T*)it->second.lock_ptr;
  }

  /** \brief Unlocks a block of memory previously locked by lock_ro()
          (does nothing in this implementation)
      \param p Pointer to the block of memory.
   **/
  static void unlock_ro(pointer_type p) {
    if (xm_allocator_get_path(alloc_data::get_instance().xm_allocator_inst) == NULL)
      return;

    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);
    map_type::iterator it = alloc_data::get_instance().g_map.find(p);
    if (it == alloc_data::get_instance().g_map.end())
      throw std::runtime_error("unlock_ro: pointer not allocated");
    if (it->second.lock_ptr == NULL)
      throw std::runtime_error("unlock_ro: pointer not locked");
    if (it->second.n_locks == 0) throw std::runtime_error("unlock_ro: block not locked");
    it->second.n_locks--;
    if (it->second.n_locks == 0) {
      free(it->second.lock_ptr);
      it->second.lock_ptr = NULL;
    }
  }

  /** \brief Locks a block of memory in physical space for read-write
          (does nothing in this implementation)
      \param p Pointer to the block of memory.
      \return Physical pointer to the memory.
   **/
  static T* lock_rw(pointer_type p) { return (T*)lock_ro(p); }

  /** \brief Unlocks a block of memory previously locked by lock_rw()
          (does nothing in this implementation)
      \param p Pointer to the block of memory.
   **/
  static void unlock_rw(pointer_type p) {
    if (xm_allocator_get_path(alloc_data::get_instance().xm_allocator_inst) == NULL)
      return;

    libutil::auto_lock<libutil::mutex> lock(alloc_data::get_instance().g_lock);
    map_type::iterator it = alloc_data::get_instance().g_map.find(p);
    if (it == alloc_data::get_instance().g_map.end())
      throw std::runtime_error("unlock_rw: pointer not allocated");
    if (it->second.lock_ptr == NULL)
      throw std::runtime_error("unlock_rw: pointer not locked");
    if (it->second.n_locks == 0) throw std::runtime_error("unlock_rw: block not locked");
    it->second.n_locks--;
    if (it->second.n_locks == 0) {
      xm_allocator_write(alloc_data::get_instance().xm_allocator_inst, p,
                         it->second.lock_ptr, it->second.size_bytes);
      free(it->second.lock_ptr);
      it->second.lock_ptr = NULL;
    }
  }

  /** \brief Sets a priority flag on a memory block (stub)
      \param p Pointer to a block of memory.
   **/
  static void set_priority(pointer_type p) {}

  /** \brief Unsets a priority flag on a memory block (stub)
      \param p Pointer to a block of memory.
   **/
  static void unset_priority(pointer_type p) {}
};

template <typename T>
const typename lt_xm_allocator<T>::pointer_type lt_xm_allocator<T>::invalid_pointer =
      XM_NULL_PTR;

}  // namespace lt_xm_allocator
}  // namespace libtensor

#endif  // LIBTENSOR_XM_ALLOCATOR_H
