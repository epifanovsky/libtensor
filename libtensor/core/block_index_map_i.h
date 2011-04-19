#ifndef LIBTENSOR_BLOCK_INDEX_MAP_I_H
#define LIBTENSOR_BLOCK_INDEX_MAP_I_H

#include "block_index_space.h"

namespace libtensor {


/**	\brief Interface to block %index mappings
	\tparam N Tensor order.

	Block %index maps provide mappings from one block %index space to
	another. The source and destination spaces are returned by
	get_bis_from() and get_bis_to();

	get_map() shall be used to obtain mappings for individual block
	indexes. The mapping procedure shall be deterministic and dependent
	only on the source block %index. The source and destination block
	indexes shall be valid block %indexes in the respective spaces.
	The %dimensions of mapped blocks must be identical for the mapping
	to be valid. Invalid source block indexes supplied to get_map()
	shall raise an exception.

	\ingroup libtensor_core
 **/
template<size_t N>
class block_index_map_i {
public:
	/**	\brief Virtual destructor
	 **/
	virtual ~block_index_map_i() { }

	/**	\brief Returns the source block %index space
	 **/
	virtual const block_index_space<N> &get_bis_from() const = 0;

	/**	\brief Returns the destination block %index space
	 **/
	virtual const block_index_space<N> &get_bis_to() const = 0;

	/**	\brief Returns true if a mapping from the given source
			block %index exists, false otherwise
	 **/
	virtual bool map_exists(const index<N> &from) const = 0;

	/**	\brief Returns the mapping from source block %index to
			destination block %index
		\param from Source %index.
		\param[out] to Destination %index.
	 **/
	virtual void get_map(const index<N> &from, index<N> &to) const = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_MAP_I_H
