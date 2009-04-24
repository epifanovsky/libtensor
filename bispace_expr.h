#ifndef LIBTENSOR_BISPACE_EXPR_H
#define	LIBTENSOR_BISPACE_EXPR_H

#include "defs.h"
#include "exception.h"

/**	\defgroup libtensor_bispace_expr Block %index space expressions
	\ingroup libtensor

	The members of this group provide the facility to create block %index
	spaces with arbitrary symmetry.
 **/

namespace libtensor {

/**	\brief Base class for block %index space expressions
	\tparam N Expression order

	\ingroup libtensor_bispace_expr
 **/
template<size_t N>
class bispace_expr_base {
};

/**	\brief Block %index space expression
	\tparam N Expression order
	\tparam T Underlying expression type

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename T>
class bispace_expr {
};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_H

