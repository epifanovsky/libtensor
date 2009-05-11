#ifndef LIBTENSOR_BISPACE_I_H
#define	LIBTENSOR_BISPACE_I_H

#include "defs.h"
#include "exception.h"
#include "block_index_space_i.h"
#include "ispace_i.h"
#include "rc_ptr.h"

namespace libtensor {

/**	\brief Block %index space interface
	\tparam N Index space order

	\ingroup libtensor
 **/
template<size_t N>
class bispace_i : public block_index_space_i<N> {
public:
	/**	\brief Creates an identical copy of the %index space
	 **/
	virtual rc_ptr<bispace_i<N> > clone() const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_I_H

