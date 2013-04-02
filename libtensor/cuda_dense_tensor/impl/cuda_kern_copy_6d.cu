#include "cuda_kern_copy_6d.h"
#include <libtensor/cuda/cuda_utils.h>

namespace libtensor {


const char *cuda_kern_copy_6d::k_clazz = "cuda_kern_copy_6d";

cuda_kern_copy_6d::cuda_kern_copy_6d(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<6> dimsa, const permutation<6> &perma, const double &c, const double &d) :
		cuda_kern_copy_generic(pa, pb, c, d) {
	//create a simple sequence map and permute it to get coefficients in the permuted tensor
	sequence<6, size_t>  map(0);
	for (int i=0; i < 4 ; i++) {
		map[i] =i;
	}
	perma.apply(map);

//	//get incriments using the map
	b_incrs1 = make_uint3(dimsa.get_increment(map[0]), dimsa.get_increment(map[1]), dimsa.get_increment(map[2]));
	b_incrs2 = make_uint3(dimsa.get_increment(map[3]), dimsa.get_increment(map[4]), dimsa.get_increment(map[5]));

	// setup execution parameters
	threads.x = dimsa.get_dim(0);
	threads.y = dimsa.get_dim(1);
	threads.z = dimsa.get_dim(2);
	grid.x = dimsa.get_dim(3);
	grid.y = dimsa.get_dim(4);
	grid.z = dimsa.get_dim(5);
}

void cuda_kern_copy_6d::run() {
	static const char *method =
			        "run( )";
//	//kernel call
	if (m_d != 0) {
   			cuda::add_copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs1, b_incrs2, dims1, m_c*m_d);
   	} else {
   		if (m_c == 1) {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs1, b_incrs2, dims1);
   		} else {
   			cuda::copy_tensor<<<grid, threads>>>(m_pa.get_physical_pointer(), m_pb.get_physical_pointer(), b_incrs1, b_incrs2, dims1, m_c);
   		}
   	}
   	cuda_utils::handle_kernel_error(g_ns, k_clazz, method, __FILE__, __LINE__);
}


} // namespace libtensor
