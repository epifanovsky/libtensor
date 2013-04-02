#include "cuda_kern_copy_4d.h"
#include <libtensor/cuda/cuda_utils.h>

namespace libtensor {


const char *cuda_kern_copy_4d::k_clazz = "cuda_kern_copy_4d";

cuda_kern_copy_4d::cuda_kern_copy_4d(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<4> dimsa, const permutation<4> &perma, const double &c, const double &d) :
		cuda_kern_copy_generic(pa, pb, c, d) {
		//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		const size_t N = 4;
		sequence<4, size_t>  map(0);
		for (size_t i = 0; i < 4 ; i++) {
			map[i] =i;
		}
		perma.apply(map);

		//get b increments using the map
		b_incrs = make_uint4(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]), dimsa.get_increment(map[2]), dimsa.get_increment(map[3]));

		// setup execution parameters
		threads.x = dimsa.get_dim(0);
		threads.y = dimsa.get_dim(1);
	  	grid.x = dimsa.get_dim(2);
	  	grid.y = dimsa.get_dim(3);
	}

void cuda_kern_copy_4d::run() {

	static const char *method =
			        "run( )";
	//kernel call
   	if (m_d != 0) {
   			cuda::add_copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs, dims, m_c*m_d);
   	} else {
   		if (m_c == 1) {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs, dims);
   		} else {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs, dims, m_c);
   		}
   	}
   	cuda_utils::handle_kernel_error(g_ns, k_clazz, method, __FILE__, __LINE__);
}

//template<size_t N>
//cuda_kern_copy_generic<N> *cuda_kern_copy_4d::match(const double *pa, double *pb, const dimensions<N> &dimsa,
//		const permutation<N> &perma, const double &c) {
//
//	if (N != 4)
//	{
//		return 0;
//	} else {
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
//}
//
//template cuda_kern_copy_generic<2> *cuda_kern_copy_4d::match(const double *, double *, const dimensions<2> &,
//		const permutation<2> &, const double &);
//template cuda_kern_copy_generic<4> *cuda_kern_copy_4d::match(const double *, double *, const dimensions<4> &,
//		const permutation<4> &, const double &);
//template cuda_kern_copy_generic<6> *cuda_kern_copy_4d::match(const double *, double *, const dimensions<6> &,
//		const permutation<6> &, const double &);

} // namespace libtensor
