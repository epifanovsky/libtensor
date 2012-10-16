#ifndef LIBTENSOR_CUDA_KERN_COPY_2D
#define LIBTENSOR_CUDA_KERN_COPY_2D

#include "cuda_kern_functions.h"
#include "cuda_kern_copy_generic.h"
#include "../../core/dimensions.h"
#include "../../core/permutation.h"


namespace libtensor {


/**	\brief Kernel for \f$ b_i += c*a_i \f$

 	\ingroup libtensor_tod_kernel
 **/
class cuda_kern_copy_2d : public cuda_kern_copy_generic {
public:
	static const char *k_clazz; //!< Kernel name

private:

	uint2 b_incrs;
	dim3 threads, grid;


public:
	cuda_kern_copy_2d(const double *pa, double *pb, const dimensions<2> dimsa, const permutation<2> &perma, const double &c, const double &d);

	cuda_kern_copy_2d(const double *pa, double *pb, dim3 p_threads, dim3 p_grids, uint2 p_b_incrs, const double &c, const double &d):
		cuda_kern_copy_generic(pa, pb, c, d), threads(p_threads), grid(p_grids), b_incrs(p_b_incrs) {
	}

	virtual ~cuda_kern_copy_2d() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	/**	\brief Runs the kernel
			 **/
	virtual void run();

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_KERN_COPY_2D


