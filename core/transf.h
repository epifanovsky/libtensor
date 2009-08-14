#ifndef LIBTENSOR_TRANSF_H
#define LIBTENSOR_TRANSF_H

#include "defs.h"
#include "exception.h"

namespace libtensor {


/**	\brief Describes how the the canonical block needs to be transformed to
		 obtain a replica
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This template is a structure placeholder. It is to be specialized for
	each %tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
struct transf {
public:
	void reset() { };
};


} // namespace libtensor

#endif // LIBTENSOR_TRANSF_H
