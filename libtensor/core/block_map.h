#ifndef LIBTENSOR_BLOCK_MAP_H
#define LIBTENSOR_BLOCK_MAP_H

#include <map>
#include "../defs.h"
#include "../exception.h"
#include "dimensions.h"
#include "immutable.h"
#include "../mp/mp_safe_tensor.h"

namespace libtensor {

/** \brief Stores pointers to block as an associative array
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Allocator used to obtain memory for tensors.

    The block map is an associative array of blocks represented by %tensor
    objects with absolute indexes as keys. This class maintains such a map
    and provides facility to create and remove blocks. All the necessary
    memory management is done here as well.

    \ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc>
class block_map : public immutable {
public:
    typedef mp_safe_tensor<N, T, Alloc> tensor_t;
    typedef std::pair<size_t, tensor_t*> pair_t;
    typedef std::map<size_t, tensor_t*> map_t;

private:
    static const char *k_clazz; //!< Class name

public:
    map_t m_map; //!< Map that stores all the pointers

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Constructs the map
     **/
    block_map();

    /** \brief Destroys the map and all the blocks
     **/
    ~block_map();

    //@}

    //!    \name Operations with blocks
    //@{

    /** \brief Creates a block with given %index and %dimensions. If
            the block exists, it is removed and re-created
        \param idx Absolute %index of the block.
        \param dims Dimensions of the block.
        \throw immut_violation If the object is immutable.
        \throw out_of_memory If there is not enough memory to create
            the block.
     **/
    void create(size_t idx, const dimensions<N> &dims)
        throw(immut_violation, out_of_memory);

    /** \brief Removes a block
        \param idx Absolute %index of the block.
        \throw immut_violation If the object is immutable.
     **/
    void remove(size_t idx) throw(immut_violation);

    /** \brief Returns whether a block with a given %index exists
        \param idx Absolute %index of the block.
     **/
    bool contains(size_t idx) const;

    /** \brief Returns the reference to a block identified by the
            %index
        \param idx Absolute %index of the block
        \throw block_not_found If the %index supplied does not
            correspond to a block
     **/
    dense_tensor<N, T, Alloc> &get(size_t idx) throw(block_not_found);

    /** \brief Removes all blocks
        \throw immut_violation If the object is immutable.
     **/
    void clear() throw(immut_violation);

    //@}

protected:
    //!    \name Implementation of immutable
    //@{

    virtual void on_set_immutable();

    //@}

private:
    /** \brief Removes all blocks (without checking for immutability)
     **/
    void do_clear();

};


template<size_t N, typename T, typename Alloc>
const char *block_map<N, T, Alloc>::k_clazz = "block_map<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
block_map<N, T, Alloc>::block_map() {

}


template<size_t N, typename T, typename Alloc>
block_map<N, T, Alloc>::~block_map() {

    do_clear();
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::create(size_t idx, const dimensions<N> &dims)
    throw(out_of_memory, immut_violation) {

    static const char *method = "create(size_t, const dimensions<N>&)";

    if(is_immutable()) {
        throw immut_violation("libtensor", k_clazz, method, __FILE__,
            __LINE__, "Immutable object cannot be modified.");
    }

    try {
        tensor_t *ptr = new mp_safe_tensor<N, T, Alloc>(dims);

        typename map_t::iterator i = m_map.find(idx);
        if(i == m_map.end()) {
            m_map.insert(pair_t(idx, ptr));
        } else {
            delete i->second;
            i->second = ptr;
        }
    } catch(std::bad_alloc &e) {
        throw out_of_memory("libtensor", k_clazz, method, __FILE__,
            __LINE__, "Not enough memory to create a block.");
    }
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::remove(size_t idx) throw(immut_violation) {

    static const char *method = "remove(size_t)";

    if(is_immutable()) {
        throw immut_violation("libtensor", k_clazz, method, __FILE__,
            __LINE__, "Immutable object cannot be modified.");
    }

    typename map_t::iterator i = m_map.find(idx);
    if(i != m_map.end()) {
        delete i->second;
        m_map.erase(i);
    }
}


template<size_t N, typename T, typename Alloc>
inline bool block_map<N, T, Alloc>::contains(size_t idx) const {

    return m_map.find(idx) != m_map.end();
}


template<size_t N, typename T, typename Alloc>
dense_tensor<N, T, Alloc> &block_map<N, T, Alloc>::get(size_t idx)
    throw(block_not_found) {

    static const char *method = "get(size_t)";

    typename map_t::iterator i = m_map.find(idx);
    if(i == m_map.end()) {
        throw block_not_found("libtensor", k_clazz, method, __FILE__,
            __LINE__, "Requested block cannot be located.");
    }

    return *(i->second);
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::clear() throw(immut_violation) {

    static const char *method = "clear()";

    if(is_immutable()) {
        throw immut_violation("libtensor", k_clazz, method, __FILE__,
            __LINE__, "Immutable object cannot be modified.");
    }

    do_clear();
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::on_set_immutable() {

    typename map_t::iterator i = m_map.begin();
    while(i != m_map.end()) {
        i->second->set_immutable();
        i++;
    }
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::do_clear() {

    typename map_t::iterator i = m_map.begin();
    while(i != m_map.end()) {
        tensor_t *ptr = i->second;
        i->second = NULL;
        delete ptr;
        i++;
    }
    m_map.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_H
