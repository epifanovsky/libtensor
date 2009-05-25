#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_H

#include "defs.h"
#include "exception.h"
#include "block_index_space_i.h"

namespace libtensor {

/**	\brief Block index space
	\tparam N Tensor order.

	\ingroup libtensor
 **/
template<size_t N>
class block_index_space : public block_index_space_i<N> {
private:
	dimensions<N> m_dims; //!< Total dimensions

public:
	//!	\name Construction and destruction
	//@{
	block_index_space(const dimensions<N> &dims);
	block_index_space(const block_index_space<N> &bis);
	virtual ~block_index_space();
	//@}

	//!	\name Implementation of libtensor::block_index_space_i<N>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

	void permute(const permutation<N> &perm);
};

template<size_t N>
block_index_space<N>::block_index_space(const dimensions<N> &dims) :
	m_dims(dims) {
}

template<size_t N>
block_index_space<N>::block_index_space(const block_index_space<N> &bis) :
	m_dims(bis.dims) {
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
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SPACE_H
