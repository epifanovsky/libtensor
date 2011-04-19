#ifndef LIBTENSOR_BTENSOR_TRAITS_H
#define LIBTENSOR_BTENSOR_TRAITS_H

#include <libvmm/ec_allocator.h>
#include <libvmm/vm_allocator.h>
#include <libvmm/std_allocator.h>

namespace libtensor {


template<typename T>
struct btensor_traits {
	typedef T element_t;
#ifdef LIBTENSOR_DEBUG
	typedef libvmm::ec_allocator< T, libvmm::vm_allocator<T>,
		libvmm::std_allocator<T> > allocator_t;
#else // LIBTENSOR_DEBUG
	typedef libvmm::vm_allocator<T> allocator_t;
#endif // LIBTENSOR_DEBUG
};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_TRAITS_H
