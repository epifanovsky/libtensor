#ifndef LIBTENSOR_INDEX_SPACE_PRODUCT_H
#define LIBTENSOR_INDEX_SPACE_PRODUCT_H

#include "defs.h"
#include "exception.h"
#include "index_product.h"

namespace libtensor {

/**	\brief Direct product of two %index spaces
	\param IS1 First %index space
	\param IS1 Second %index space
**/
template<typename IS1, typename IS2>
class index_space_product {
public:
	typedef index_product<typename IS1::index_t, typename IS2::index_t>
		index_t; //!< Index type
};

} // namespace libtensor

#endif // LIBTENSOR_INDEX_SPACE_PRODUCT_H

