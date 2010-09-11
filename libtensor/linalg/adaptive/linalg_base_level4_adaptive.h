#ifndef LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H
#define LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H

#include "../generic/linalg_base_level4_generic.h"

namespace libtensor {


/**	\brief Level-4 linear algebra operations (adaptive)
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.
	\tparam L3 Level-3 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_level4_adaptive : public linalg_base_level4_generic {

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_LEVEL4_ADAPTIVE_H
