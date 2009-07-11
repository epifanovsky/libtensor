#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "block_index_space_i.h"
#include "dimensions.h"
#include "index.h"
#include "permutation.h"

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

	\ingroup libtensor
 **/
template<size_t N>
class block_index_space : public block_index_space_i<N> {
private:
	static const char *k_clazz; //!< Class name

private:
	dimensions<N> m_dims; //!< Total dimensions
	index<N> m_block_index_max; //!< The last block %index
	std::list< index<N> > m_splits; //!< Split points

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

	/**	\brief Virtual destructor
	 **/
	virtual ~block_index_space();

	//@}


	//!	\name Dimensions of the block %index space
	//@{

	/**	\brief Returns the total dimensions
	 **/
	virtual const dimensions<N> &get_dims() const;

	/**	\brief Returns the dimensions that limit block %index values
	 **/
	dimensions<N> get_block_index_dims() const;

	/**	\brief Returns the dimensions of a block
		\param idx Block index.
		\throw exception If the index is out of bounds.
	 **/
	dimensions<N> get_block_dims(const index<N> &idx) const
		throw(exception);

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Adds a split point for a dimension
		\param dim Dimension number (not to exceed N).
		\param pos Split position (not to exceed the number of
			elements along the given dimension).
		\throw exception If one of the indexes is out of bounds.
	 **/
	void split(size_t dim, size_t pos) throw(exception);

	/**	\brief Removes all split points
	 **/
	void reset();

	/**	\brief Permutes the block %index space
		\param perm Permutation.
	 **/
	void permute(const permutation<N> &perm);

	//@}

};

template<size_t N>
const char *block_index_space<N>::k_clazz = "block_index_space<N>";

template<size_t N>
block_index_space<N>::block_index_space(const dimensions<N> &dims)
: m_dims(dims) {

	index<N> idx;
	for(register size_t i = 0; i < N; i++) idx[i] = dims[i];
	m_splits.push_back(idx);
}

template<size_t N>
block_index_space<N>::block_index_space(const block_index_space<N> &bis)
: m_dims(bis.m_dims), m_block_index_max(bis.m_block_index_max),
	m_splits(bis.m_splits) {

}

template<size_t N>
block_index_space<N>::~block_index_space() {

}

template<size_t N>
inline const dimensions<N> &block_index_space<N>::get_dims() const {

	return m_dims;
}

template<size_t N>
inline void block_index_space<N>::permute(const permutation<N> &perm) {

	m_dims.permute(perm);
	m_block_index_max.permute(perm);
	typename std::list< index<N> >::iterator iter = m_splits.begin();
	while(iter != m_splits.end()) {
		iter->permute(perm);
		iter++;
	}
}

template<size_t N>
void block_index_space<N>::split(size_t dim, size_t pos) throw(exception) {

	static const char *method = "split(size_t, size_t)";

	if(dim >= N)
		throw_exc(k_clazz, method, "dim is out of bounds");
	if(pos >= m_dims[dim])
		throw_exc(k_clazz, method, "pos is out of bounds");
	if(pos == 0)
		return;

	index<N> i0;
	typename std::list< index<N> >::iterator iter = m_splits.begin();
	typename std::list< index<N> >::iterator ins = m_splits.end();
	index<N> last;
	do {
		index<N> &cur = *iter;

		if(cur[dim] == pos) break;
		if(cur[dim] == last[dim]) ins = iter;
		if(cur[dim] > pos) {
			if(ins == m_splits.end()) {
				index<N> i(last);
				i[dim] = pos;
				m_splits.insert(iter, i);
			} else {
				while(ins != iter) {
					index<N> &i1 = *ins;
					ins++;
					index<N> &i2 = *ins;
					if(ins == iter)
						i1[dim] = pos;
					else
						i1[dim] = i2[dim];
				}
			}
			m_block_index_max[dim]++;
			break;
		}

		last = *iter;
		iter++;
	} while(iter != m_splits.end());
}

template<size_t N>
void block_index_space<N>::reset() {

	index<N> idx;
	for(register size_t i = 0; i < N; i++) {
		m_block_index_max[i] = 0;
		idx[i] = m_dims[i];
	}
	m_splits.clear();
	m_splits.push_back(idx);
}

template<size_t N>
dimensions<N> block_index_space<N>::get_block_index_dims() const {

	index<N> i0;
	return dimensions<N>(index_range<N>(i0, m_block_index_max));
}

template<size_t N>
dimensions<N> block_index_space<N>::get_block_dims(const index<N> &idx) const
	throw(exception) {

	index<N> i1, i2;
	typename std::list< index<N> >::const_iterator iter = m_splits.begin();
	index<N> last, blk;
	do {
		const index<N> &cur = *iter;
		for(size_t i = 0; i < N; i++) {
			if(cur[i] > last[i]) {
				if(blk[i] == idx[i]) {
					i1[i] = last[i];
					i2[i] = cur[i] - 1;
				}
				blk[i]++;
			}
		}
		last = *iter;
		iter++;
	} while(iter != m_splits.end());

	return dimensions<N>(index_range<N>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_H
