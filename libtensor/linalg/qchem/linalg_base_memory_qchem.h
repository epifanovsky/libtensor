#ifndef LIBTENSOR_LINALG_BASE_MEMORY_QCHEM_H
#define LIBTENSOR_LINALG_BASE_MEMORY_QCHEM_H

#include <libvmm/vm_fast_buffer.h>

namespace libtensor {


/**	\brief Memory buffer allocation (generic)

	\ingroup libtensor_linalg
 **/
struct linalg_base_memory_generic {

	/**	\brief Allocates a temporary array of doubles
		\param n Array length.
		\return Pointer to the array.
	 **/
	static double *allocate(size_t n) {
		return vm_fast_buffer<double>::allocate(n);
	}

	/**	\brief Deallocates a temporary array previously allocated
			using allocate(size_t)
		\param p Pointer to the array.
	 **/
	static void deallocate(double *p) {
		vm_fast_buffer<double>::deallocate(p);
	}

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_MEMORY_QCHEM_H
