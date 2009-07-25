#ifndef LIBTENSOR_BLOCK_MAP_H
#define LIBTENSOR_BLOCK_MAP_H

#include <map>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "tensor.h"

namespace libtensor {

/**	\brief Stores pointers to block as an associative array
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
class block_map {
public:
	typedef tensor<N, T, Alloc> tensor_t;
	typedef std::pair<size_t, tensor_t*> pair_t;
	typedef std::map<size_t, tensor_t*> map_t;

private:
	static const char *k_clazz; //!< Class name

public:
	map_t m_map; //!< Map that stores all the pointers

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructs the map
	 **/
	block_map();

	/**	\brief Destroys the map and all the blocks
	 **/
	~block_map();

	//@}

	//!	\name Operations with blocks
	//@{

	/**	\brief Creates a block with given %index and %dimensions. If
			the block exists, it is removed and re-created
		\param idx Absolute %index of the block.
		\param dims Dimensions of the block.
		\throw out_of_memory If there is not enough memory to create
			the block.
	 **/
	void create(size_t idx, const dimensions<N> &dims) throw(out_of_memory);

	/**	\brief Removes a block
		\param idx Absolute %index of the block.
	 **/
	void remove(size_t idx);

	/**	\brief Returns whether a block with a given %index exists
		\param idx Absolute %index of the block.
	 **/
	bool contains(size_t idx);

	/**	\brief Returns the reference to a block identified by the
			%index
		\param idx Absolute %index of the block
		\throw block_not_found If the %index supplied does not
			correspond to a block
	 **/
	tensor<N, T, Alloc> &get(size_t idx) throw(block_not_found);

	/**	\brief Removes all blocks
	 **/
	void clear();

	//@}

};


template<size_t N, typename T, typename Alloc>
const char *block_map<N, T, Alloc>::k_clazz = "block_map<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
block_map<N, T, Alloc>::block_map() {

}


template<size_t N, typename T, typename Alloc>
block_map<N, T, Alloc>::~block_map() {

	clear();
}


template<size_t N, typename T, typename Alloc>
void block_map<N, T, Alloc>::create(size_t idx, const dimensions<N> &dims)
	throw(out_of_memory) {

	static const char *method = "create(size_t, const dimensions<N>&)";

	try {
		tensor_t *ptr = new tensor<N, T, Alloc>(dims);

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
void block_map<N, T, Alloc>::remove(size_t idx) {

	typename map_t::iterator i = m_map.find(idx);
	if(i != m_map.end()) {
		delete i->second;
		m_map.erase(i);
	}
}


template<size_t N, typename T, typename Alloc>
inline bool block_map<N, T, Alloc>::contains(size_t idx) {

	return m_map.find(idx) != m_map.end();
}


template<size_t N, typename T, typename Alloc>
tensor<N, T, Alloc> &block_map<N, T, Alloc>::get(size_t idx)
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
void block_map<N, T, Alloc>::clear() {

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
