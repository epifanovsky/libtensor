#ifndef LIBTENSOR_CUDA_KERN_COPY_GENERIC_H
#define LIBTENSOR_CUDA_KERN_COPY_GENERIC_H

//#include "kernel_base.h"
#include "../../core/dimensions.h"
#include "../../core/permutation.h"

namespace libtensor {


/** \brief Generic kernel for copy

 	\ingroup libtensor_tod_kernel
 **/
//template<size_t N>
class cuda_kern_copy_generic {
//	friend class cuda_kern_copy_4d;
//	friend class cuda_kern_copy_6d;

public:
	static const char *k_clazz; //!< Kernel name

protected:
//	size_t m_size; //!< Total size of the tensor
//	size_t fist_dim;
//	size_t second_dim;

	const double *m_pa;
	double *m_pb;
	const double m_c; //scaled factor
	const double m_d; //scaled factor for addition
//	size_t dim1, dim2, size;
//private:
//	const dimensions<N> m_dimsa;

public:
//	cuda_kern_copy_generic(const double *pa, double *pb, const dimensions<N> &dimsa, const double &c) :
//				m_pa(pa), m_pb(pb), m_dimsa(dimsa), m_c(c) {
//		}

	cuda_kern_copy_generic(const double *pa, double *pb, const double &c, const double &d) :
					m_pa(pa), m_pb(pb), m_c(c), m_d(d) {
			}



	virtual ~cuda_kern_copy_generic() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	/**	\brief Runs the kernel
		 **/

//	virtual void run() = 0;

	virtual void run();
//	{
//		//do nothing
//	}

	template<size_t N>
	static cuda_kern_copy_generic *match(const double *pa, double *pb, const dimensions<N> &dimsa, const permutation<N> &perma, const double &c, const double &d);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_KERN_COPY_GENERIC_H
