#ifndef LIBTENSOR_LINALG_BASE_GENERIC_H
#define LIBTENSOR_LINALG_BASE_GENERIC_H

#include "linalg_base_lowlevel.h"
#include "linalg_base_memory_generic.h"
#include "linalg_base_level1_generic.h"
#include "linalg_base_level2_generic.h"
#include "linalg_base_level3_generic.h"
#include "linalg_base_level4_generic.h"
#include "linalg_base_level5_generic.h"
#include "linalg_base_level6_generic.h"

namespace libtensor {


/**	\brief Generic linear algebra implementation

	\ingroup libtensor_linalg
 **/
struct linalg_base_generic :
	public linalg_base_lowlevel<
		linalg_base_memory_generic,
		linalg_base_level1_generic,
		linalg_base_level2_generic,
		linalg_base_level3_generic>,
	public linalg_base_level4_generic,
	public linalg_base_level5_generic,
	public linalg_base_level6_generic
{ };


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_GENERIC_H
