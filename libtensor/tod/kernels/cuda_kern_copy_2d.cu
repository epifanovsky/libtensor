#include "cuda_kern_copy_2d.h"

namespace libtensor {


const char *cuda_kern_copy_2d::k_clazz = "cuda_kern_copy_2d";

cuda_kern_copy_2d::cuda_kern_copy_2d(const double *pa, double *pb, const dimensions<2> dimsa, const dimensions<2> dimsb, const permutation<2> &perma, const double &c, const double &d) :
	cuda_kern_copy_generic(pa, pb, c, d) {

	const int THREADS_PER_BLOCK = 512;
	//create a simple sequence map and permute it to get coefficients in the permuted tensor
//		const size_t N = 4;
	sequence<2, size_t>  map(0);
	for (size_t i = 0; i < 2 ; i++) {
		map[i] =i;
	}
	perma.apply(map);

	//get b increments using the map
	b_incrs = make_uint2(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]));

	dims = make_uint2(dimsa.get_dim(0), dimsa.get_dim(1));

	// setup execution parameters
	threads.x = (dimsa.get_dim(1) < THREADS_PER_BLOCK) ? dimsa.get_dim(1) : THREADS_PER_BLOCK;
  	grid.x = dimsa.get_dim(0);
}

void cuda_kern_copy_2d::run() {

	//kernel call
   	if (m_d != 0) {
   			cuda::add_copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs, dims, m_c*m_d);
   	} else {
   		if (m_c == 1) {
//   			std::cout << "\nb_inc = " << b_incrs.x << " " << b_incrs.y << "\n";
//   			std::cout << "\dims = " << dims.x << " " << dims.y << "\n";
   			cuda::copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs, dims);
   		} else {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa, m_pb, b_incrs, dims, m_c);
   		}
   	}
}
} // namespace libtensor
