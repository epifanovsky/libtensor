#ifndef LIBTENSOR_SO_COPY_H
#define LIBTENSOR_SO_COPY_H

#include "defs.h"
#include "exception.h"
#include "symmetry_operation_i.h"
#include "symmetry_operation_target.h"

namespace libtensor {

template<size_t N, typename T>
class so_copy : public symmetry_operation_i <N, T>,
	public symmetry_operation_target< N, T, perm_symmetry<N, T> > {

public:
	void perform(perm_symmetry<N, T> &sym) throw(exception) {

	}
};

} // namespace libtensor

#endif // LIBTENSOR_SO_COPY_H
