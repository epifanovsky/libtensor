#include "cuda_kern_copy_2d.h"

namespace libtensor {


const char *cuda_kern_copy_2d::k_clazz = "cuda_kern_copy_2d";

cuda_kern_copy_2d::cuda_kern_copy_2d(const double *pa, double *pb, const dimensions<2> dimsa, const permutation<2> &perma, const double &c, const double &d) :
	cuda_kern_copy_generic(pa, pb, c, d) {
	//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		const size_t N = 4;
	sequence<2, size_t>  map(0);
	for (size_t i = 0; i < 2 ; i++) {
		map[i] =i;
	}
	perma.apply(map);

	//get b increments using the map
	b_incrs = make_uint2(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]));

	// setup execution parameters
	threads.x = dimsa.get_dim(0);
  	grid.x = dimsa.get_dim(1);
}

void cuda_kern_copy_2d::run() {

	//kernel call
   	if (m_d != 0) {
   			cuda::add_copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs, m_c*m_d);
   	} else {
   		if (m_c == 1) {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs);
   		} else {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs, m_c);
   		}
   	}
}
} // namespace libtensor
