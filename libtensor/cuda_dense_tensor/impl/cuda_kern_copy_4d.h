#ifndef LIBTENSOR_CUDA_KERN_COPY_4D
#define LIBTENSOR_CUDA_KERN_COPY_4D

#include "cuda_kern_functions.h"
#include "cuda_kern_copy_generic.h"
//#include "../../core/sequence.h"
#include "../../core/dimensions.h"
#include "../../core/permutation.h"


namespace libtensor {


/**	\brief Kernel for \f$ b_i += c*a_i \f$

 	\ingroup libtensor_tod_kernel
 **/
//template<size_t N>
class cuda_kern_copy_4d : public cuda_kern_copy_generic {
public:
	static const char *k_clazz; //!< Kernel name

private:
//	double *m_pa, *m_pb;
//	dimensions<4> m_dimsa;
//	double m_c;

	uint4 b_incrs, dims;
	dim3 threads, grid;


public:
	cuda_kern_copy_4d(const double *pa, double *pb, const dimensions<4> dimsa, const permutation<4> &perma, const double &c, const double &d);

	cuda_kern_copy_4d(const double *pa, double *pb, dim3 p_threads, dim3 p_grids, uint4 p_b_incrs, uint4 p_dims, const double &c, const double &d):
		cuda_kern_copy_generic(pa, pb, c, d), threads(p_threads), grid(p_grids), b_incrs(p_b_incrs), dims(p_dims) {
	}
//	cuda_kern_copy_generic<4>(pa, pb, dimsa, c) {
//		//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		const size_t N = 4;
//		sequence<N, size_t>  map(0);
//		size_t t = dimsa.get_increment(map[0]);
//		for (size_t i = 0; i < 4 ; i++) {
//			map[i] =i;
//		}
//		perma.apply(map);
//
//		//get b increments using the map
//		b_incrs = make_uint4(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]), dimsa.get_increment(map[2]), dimsa.get_increment(map[3]));
//	}

	virtual ~cuda_kern_copy_4d() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	/**	\brief Runs the kernel
			 **/
	virtual void run();

//	template<size_t N>
//	static cuda_kern_copy_generic<N> *match(const double *pa, double *pb, const dimensions<N> &dimsa, const permutation<N> &perma, const double &c);
//	{
//
//		if (N != 4) return 0;
//
//		cuda_kern_copy_4d zz(pa, pb, dimsa, c);
//	//	zz.pa = pa;
//	//	zz.pb = pb;
//	//	zz.c = c;
//
//		//save dimensions of a
//	//	zz.dimsa = dimsa;
//
//
//		//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		sequence<N, size_t>  map();
//		for (int i=0; i < N ; i++) {
//			map[i] =i;
//		}
//		perma.apply(map);
//
//		//get b increments using the map
//		zz.b_incrs = make_uint4(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]), dimsa.get_increment(map[2]), dimsa.get_increment(map[3]));
//
//		return new cuda_kern_copy_4d(zz);
//	}

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_KERN_COPY_4D


