#ifndef LIBTENSOR_LINALG_BASE_HIGHLEVEL_H
#define LIBTENSOR_LINALG_BASE_HIGHLEVEL_H

#include "linalg_base_level4_adaptive.h"
#include "linalg_base_level5_adaptive.h"
#include "linalg_base_level6_adaptive.h"

namespace libtensor {


/**	\brief Provides high-level (Level 4 and up) linear algebra routines
	\tparam M Memory driver.
	\tparam L1 Level-1 driver.
	\tparam L2 Level-2 driver.
	\tparam L3 Level-3 driver.

	\ingroup libtensor_linalg
 **/
template<typename M, typename L1, typename L2, typename L3>
struct linalg_base_highlevel :
	public linalg_base_level4_adaptive<M, L1, L2, L3>,
	public linalg_base_level5_adaptive<M, L1, L2, L3>,
	public linalg_base_level6_adaptive<M, L1, L2, L3>
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_HIGHLEVEL_H
