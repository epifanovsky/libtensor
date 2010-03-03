#ifndef LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H
#define LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H

#include <libvmm/singleton.h>

namespace libtensor {


template<typename OperT>
class symmetry_operation_dispatcher :
	public libvmm::singleton< symmetry_operation_dispatcher<OperT> > {

	friend class libvmm::singleton< symmetry_operation_dispatcher<OperT> >;

protected:
	symmetry_operation_dispatcher() { }
};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_OPERATION_DISPATCHER_H