#ifndef LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_H
#define LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_H

#include "block_index_space.h"

namespace libtensor {


/**	\brief Builds a block %index subspace
	\tparam N Order of the resulting space.
	\tparam M Number of removed dimensions.

	Constructs a block %index subspace given the original block %index
	space and a %mask of dimensions that will stay. If certain %dimensions
	had the same type in the original block index %space, they will also
	have the same type in the result.

	\ingroup libtensor_core
 **/
template<size_t N, size_t M>
class block_index_subspace_builder {
public:
	static const char *k_clazz; //!< Class name

private:
	block_index_space<N> m_bis; //!< Result

public:
	/**	\brief Constructs the subspace using the original space bis
			and a %mask msk.
	 **/
	block_index_subspace_builder(const block_index_space<N + M> &bis,
		const mask<N + M> &msk);

	/**	\brief Returns the subspace
	 **/
	const block_index_space<N> &get_bis() const {
		return m_bis;
	}

private:
	static dimensions<N> make_dims(const block_index_space<N + M> &bis,
		const mask<N + M> &msk);

};


template<size_t N, size_t M>
const char *block_index_subspace_builder<N, M>::k_clazz =
	"block_index_subspace_builder<N, M>";


template<size_t N, size_t M>
block_index_subspace_builder<N, M>::block_index_subspace_builder(
	const block_index_space<N + M> &bis, const mask<N + M> &msk) :

	m_bis(make_dims(bis, msk)) {

	//	At this point it is guaranteed that the mask is acceptable,
	//	since make_dims would otherwise have failed

	sequence<N, size_t> map(0);
	for(size_t i = 0, j = 0; i < N + M; i++) if(msk[i]) map[j++] = i;

	mask<N> msk_done;
	bool done = false;
	while(!done) {
		size_t i = 0;
		while(i < N && msk_done[i]) i++;
		if(i == N) {
			done = true;
			continue;
		}
		size_t typ = bis.get_type(map[i]);
		const split_points &splits = bis.get_splits(typ);
		mask<N> msk_typ;
		for(size_t j = 0; j < N; j++) {
			if(bis.get_type(map[j]) == typ) msk_typ[j] = true;
		}
		size_t npts = splits.get_num_points();
		for(size_t j = 0; j < npts; j++) {
			m_bis.split(msk_typ, splits[j]);
		}
		msk_done |= msk_typ;
	}
}


template<size_t N, size_t M>
dimensions<N> block_index_subspace_builder<N, M>::make_dims(
	const block_index_space<N + M> &bis, const mask<N + M> &msk) {

	static const char *method =
		"make_dims(const block_index_space<N + M>&, "
		"const mask<N + M>&)";

	size_t n = 0;
	for(size_t i = 0; i < N + M; i++) if(msk[i]) n++;
	if(n != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"msk");
	}

	index<N> i1, i2;
	const dimensions<N + M> &dims = bis.get_dims();
	for(size_t i = 0, j = 0; i < N + M; i++) {
		if(msk[i]) i2[j++] = dims[i] - 1;
	}

	return dimensions<N>(index_range<N>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_SUBSPACE_BUILDER_H
