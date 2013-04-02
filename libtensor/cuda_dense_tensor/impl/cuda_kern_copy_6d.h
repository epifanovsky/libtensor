#ifndef LIBTENSOR_CUDA_KERN_COPY_6D_H
#define LIBTENSOR_CUDA_KERN_COPY_6D_H

#include "cuda_kern_functions.h"
#include "cuda_kern_copy_generic.h"
#include "../../core/dimensions.h"
#include "../../core/permutation.h"

namespace libtensor {


/**	\brief Kernel for \f$ b_i += c*a_i \f$

 	\ingroup libtensor_tod_kernel
 **/
//template<size_t N>
class cuda_kern_copy_6d : public cuda_kern_copy_generic {
public:
	static const char *k_clazz; //!< Kernel name

private:
//	double *m_pa, *m_pb;
//	dimensions<6> m_dimsa;
//	double m_c;
	uint3 b_incrs1, dims1, dims2;
	uint3 b_incrs2;
	dim3 threads, grid;

public:

//	tod_dirsum(tensor_i<k_ordera, double> &ta, double ka,
//		tensor_i<k_orderb, double> &tb, double kb) :
//
//		m_ta(ta), m_tb(tb), m_ka(ka), m_kb(kb),
//		m_dimsc(mk_dimsc(ta, tb)) {
//
//
//	}

	cuda_kern_copy_6d(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<6> dimsa, const permutation<6> &perma, const double &c, const double &d);

	cuda_kern_copy_6d(cuda_pointer<const double> pa, cuda_pointer<double> pb, dim3 p_threads, dim3 p_grids, uint3 p_b_incrs1, uint3 p_b_incrs2,
											uint3 p_dims, const double &c, const double &d):
		cuda_kern_copy_generic(pa, pb, c, d), threads(p_threads), grid(p_grids), b_incrs1(p_b_incrs1), b_incrs2(p_b_incrs2),
						dims1(p_dims) {
		}

	virtual ~cuda_kern_copy_6d() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	/**	\brief Runs the kernel
			 **/

	virtual void run();

//	template<size_t N>
//	static cuda_kern_copy_generic<N> *match(const double *pa, double *pb, const dimensions<N> &dimsa, const permutation<N> &perma, const double &c);
////	{
//
//		if (N != 6) return 0;
//
//		cuda_kern_copy_6d zz(pa, pb, dimsa, c);
//	//	zz.pa = pa;
//	//	zz.pb = pb;
//	//	zz.c = c;
//	//
//	//	//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		sequence<N, size_t>  map();
//		for (int i=0; i < N ; i++) {
//			map[i] =i;
//		}
//		perma.apply(map);
//	//
//	//	//get incriments using the map
//		zz.b_incrs1 = make_uint3(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]), dimsa.get_increment(map[2]));
//		zz.b_incrs2 = make_uint3(dimsa.get_increment(map[3]), dimsa.get_increment(map[4]), dimsa.get_increment(map[5]));
//
//		return new cuda_kern_copy_6d(zz);
//	}


};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_KERN_COPY_6D_H
