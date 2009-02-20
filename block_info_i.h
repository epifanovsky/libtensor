#ifndef LIBTENSOR_BLOCK_INFO_I_H
#define LIBTENSOR_BLOCK_INFO_I_H

#include "defs.h"
#include "exception.h"
#include "dimensions.h"
#include "index.h"
#include "symmetry_i.h"

namespace libtensor {

class block_info_i {
public:
	virtual const dimensions &get_dims() const = 0;
	virtual const dimensions &get_block_dims(const index &i) const = 0;
	virtual const symmetry_i &get_symmetry() const = 0;
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INFO_I_H

