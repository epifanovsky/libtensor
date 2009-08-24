#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_H

#include <list>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "index.h"
#include "mask.h"
#include "permutation.h"
#include "split_points.h"

namespace libtensor {


/**	\brief Block %index space
	\tparam N Tensor order.

	Block %index space maintains information about the %dimensions of
	the %index space along each direction and how each subspace is split
	into individual blocks.

	A block %index space object is created using the total %dimensions
	of the %index space.

	\code
	dimensions<4> &dims = ...; // Total dimensions of the index space
	block_index_space<4> bis(dims);
	\endcode

	The newly create block %index space contains no subspace splitting
	points and therefore defines only one block that spans the entire
	%index space. Using split(), the %index space is divided into blocks.

	Dimensions split identically are suitable for permutational %symmetry
	and are said to have the same type. As the block %index space is
	initialized, those %dimensions that have the same number of elements
	are assigned the same type.

	\ingroup libtensor_core
 **/
template<size_t N>
class block_index_space {
private:
	static const char *k_clazz; //!< Class name

private:
	dimensions<N> m_dims; //!< Total dimensions
	index<N> m_nsplits; //!< Number of splits along each dimension
	sequence<N, size_t> m_type; //!< Split type
	sequence<N, split_points*> m_splits; //!< Split points

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the block %index space using given %dimensions
		\param dims Total %dimensions of the %index space
	 **/
	block_index_space(const dimensions<N> &dims);

	/**	\brief Copy constructor
		\param bis Block %index space to be cloned
	 **/
	block_index_space(const block_index_space<N> &bis);

	/**	\brief Destructor
	 **/
	~block_index_space();

	//@}


	//!	\name Dimensions of the block %index space
	//@{

	/**	\brief Returns the total dimensions
	 **/
	const dimensions<N> &get_dims() const;

	/**	\brief Returns the dimensions that limit block %index values
	 **/
	dimensions<N> get_block_index_dims() const;

	/**	\brief Returns the total %index of the first element of a block
		\param idx Block %index.
		\throw out_of_bounds If the block %index is out of bounds.
	 **/
	index<N> get_block_start(const index<N> &idx) const
		throw(out_of_bounds);

	/**	\brief Returns the dimensions of a block
		\param idx Block index.
		\throw out_of_bounds If the block %index is out of bounds.
	 **/
	dimensions<N> get_block_dims(const index<N> &idx) const
		throw(out_of_bounds);

	/**	\brief Returns the type (splitting pattern) of a dimension
		\param dim Dimension number.
		\throw out_of_bounds If the dimension number is out of bounds.
	 **/
	size_t get_type(size_t dim) const throw(out_of_bounds);

	/**	\brief Returns splitting points
		\param typ Dimension type (see get_type())
		\throw out_of_bounds If the dimension type is out of bounds.
	 **/
	const split_points &get_splits(size_t typ) const throw(out_of_bounds);

	/**	\brief Returns true if two block %index spaces are identical
	 **/
	bool equals(const block_index_space<N> &bis) const;

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Adds a split point for %dimension identified by a %mask
		\param msk Dimension mask.
		\param pos Split position (not to exceed the number of
			elements along the given dimension).
		\throw bad_parameter If the mask is incorrect.
		\throw out_of_bounds If the position is out of bounds.
	 **/
	void split(const mask<N> &msk, size_t pos)
		throw(bad_parameter, out_of_bounds);

	/**	\brief Removes all split points
	 **/
	void reset();

	/**	\brief Permutes the block %index space
		\param perm Permutation.
	 **/
	void permute(const permutation<N> &perm);

	//@}

private:
	void init_types();
	void clear_splits();
};


template<size_t N>
const char *block_index_space<N>::k_clazz = "block_index_space<N>";


template<size_t N>
block_index_space<N>::block_index_space(const dimensions<N> &dims)
: m_dims(dims), m_type(0), m_splits(NULL) {

	init_types();
}


template<size_t N>
block_index_space<N>::block_index_space(const block_index_space<N> &bis)
: m_dims(bis.m_dims), m_nsplits(bis.m_nsplits), m_type(bis.m_type),
	m_splits(NULL) {

	for(size_t i = 0; i < N; i++) {
		if(bis.m_splits[i])
			m_splits[i] = new split_points(*(bis.m_splits[i]));
	}
}


template<size_t N>
block_index_space<N>::~block_index_space() {

	clear_splits();
}

template<size_t N>
inline const dimensions<N> &block_index_space<N>::get_dims() const {

	return m_dims;
}


template<size_t N>
dimensions<N> block_index_space<N>::get_block_index_dims() const {

	index<N> i0;
	return dimensions<N>(index_range<N>(i0, m_nsplits));
}


template<size_t N>
index<N> block_index_space<N>::get_block_start(const index<N> &idx) const
	throw(out_of_bounds) {

	static const char *method = "get_block_start(const index<N>&)";

#ifdef LIBTENSOR_DEBUG
	for(register size_t i = 0; i < N; i++) {
		if(idx[i] > m_nsplits[i]) {
			throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Block index is out of bounds.");
		}
	}
#endif // LIBTENSOR_DEBUG

	index<N> i1;
	for(size_t i = 0; i < N; i++) {
		const split_points &spl = *m_splits[m_type[i]];
		if(idx[i] > 0) i1[i] = spl[idx[i] - 1];
	}

	return i1;
}


template<size_t N>
dimensions<N> block_index_space<N>::get_block_dims(const index<N> &idx) const
	throw(out_of_bounds) {

	static const char *method = "get_block_dims(const index<N>&)";

#ifdef LIBTENSOR_DEBUG
	for(register size_t i = 0; i < N; i++) {
		if(idx[i] > m_nsplits[i]) {
			throw out_of_bounds(g_ns, k_clazz, method,
				__FILE__, __LINE__,
				"Block index is out of bounds.");
		}
	}
#endif // LIBTENSOR_DEBUG

	index<N> i1, i2;
	for(size_t i = 0; i < N; i++) {
		const split_points &spl = *m_splits[m_type[i]];
		if(idx[i] > 0) i1[i] = spl[idx[i] - 1];
		i2[i] = (idx[i] == m_nsplits[i]) ?
			m_dims[i] - 1 : spl[idx[i]] - 1;
	}

	return dimensions<N>(index_range<N>(i1, i2));
}


template<size_t N>
inline size_t block_index_space<N>::get_type(size_t dim) const
	throw(out_of_bounds) {

	return m_type[dim];
}


template<size_t N>
inline const split_points &block_index_space<N>::get_splits(size_t typ) const
	throw(out_of_bounds) {

	static const char *method = "get_splits(size_t)";

	if(m_splits[typ] == NULL) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Type number is out of bounds.");
	}
	return *m_splits[typ];
}


template<size_t N>
bool block_index_space<N>::equals(const block_index_space<N> &bis) const {

	if(!m_dims.equals(bis.m_dims) || !m_nsplits.equals(bis.m_nsplits)) {
		return false;
	}

	mask<N> chk;
	for(size_t i = 0; i < N; i++) {
		size_t type1 = m_type[i], type2 = bis.m_type[i];
		if(!chk[type1]) {
			chk[type1] = true;
			for(size_t j = i + 1; j < N; j++) {
				if(m_type[j] == type1 && bis.m_type[j] != type2)
					return false;
			}
			if(!m_splits[type1]->equals(*bis.m_splits[type2]))
				return false;
		}
	}

	return true;
}


template<size_t N>
void block_index_space<N>::split(const mask<N> &msk, size_t pos)
	throw(bad_parameter, out_of_bounds) {

	static const char *method = "split(const mask<N>&, size_t)";

	size_t i, type;
	for(i = 0; i < N; i++) if(msk[i]) break;
	if(i == N) return;
	type = m_type[i];
	if(pos >= m_dims[i]) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Splitting position is out of bounds.");
	}

	mask<N> adjmsk;
	bool adjmsk_neq = false; // Whether original mask != type-adjusted mask
	for(i = 0; i < N; i++) {
		if(msk[i]) {
			if(m_type[i] != type) break;
			adjmsk[i] = true;
		} else {
			adjmsk[i] = false;
			if(m_type[i] == type) adjmsk_neq = true;
		}
	}
	if(i != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Invalid splitting mask.");
	}

	if(pos == 0) return;

	split_points *splits = NULL;
	if(adjmsk_neq) {
		size_t newtype = 0;
		for(i = 0; i < N; i++)
			if(m_type[i] > newtype) newtype = m_type[i];
		newtype++;
		m_splits[newtype] = splits =
			new split_points(*(m_splits[type]));
		for(i = 0; i < N; i++)
			if(adjmsk[i]) m_type[i] = newtype;
	} else {
		splits = m_splits[type];
	}

	if(splits->add(pos)) {
		for(i = 0; i < N; i++) if(adjmsk[i]) m_nsplits[i]++;
	}
}


template<size_t N>
void block_index_space<N>::reset() {

	clear_splits();
	init_types();
}


template<size_t N>
inline void block_index_space<N>::permute(const permutation<N> &perm) {

	m_dims.permute(perm);
	m_nsplits.permute(perm);
	m_type.permute(perm);
}


template<size_t N>
void block_index_space<N>::init_types() {

	size_t lasttype = 0;
	for(register size_t i = 0; i < N; i++) {
		size_t type = lasttype;
		for(register size_t j = 0; j < i; j++) {
			if(m_dims[i] == m_dims[j]) {
				type = m_type[j];
				break;
			}
		}
		if(type == lasttype) lasttype++;
		if(m_splits[type] == NULL)
			m_splits[type] = new split_points();
		m_type[i] = type;
	}
}


template<size_t N>
void block_index_space<N>::clear_splits() {

	for(size_t i = 0; i < N; i++) {
		delete m_splits[i];
		m_splits[i] = NULL;
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_H
