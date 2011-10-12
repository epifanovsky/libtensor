#include "cuda_kern_copy_generic.h"
#include "cuda_kern_copy_2d.h"
#include "cuda_kern_copy_4d.h"
#include "cuda_kern_copy_6d.h"
#include <iostream>

namespace libtensor {

//template<size_t N>
const char *cuda_kern_copy_generic::k_clazz = "cuda_kern_copy_generic";

//generic implementation of copy operation doesn't do permutation!
//template<size_t N>
void cuda_kern_copy_generic::run() {

	//no default copy
	//do nothing

//	// setup execution parameters
////	dim3 threads(m_dimsa.get_dim(0));
//	dim3 threads(dim1);
//	dim3 grid;
//	if (N > 1)
//	{
////		grid.x = m_dimsa.get_dim(1);
//		grid.x = dim2;
//	} else {
//		grid.x = 1;
//	}
//	//run kernel
//
//	//kernel call
//	if (m_c == 1) {
//		cuda::generic_copy_tensor<<<grid, threads>>>(m_pa, m_pb, size);
//	} else {
//		cuda::generic_add_copy_tensor<<<grid, threads>>>(m_pa, m_pb, size, m_c);
//	}
}


template<size_t N>
cuda_kern_copy_generic *cuda_kern_copy_generic::match(const double *pa, double *pb, const dimensions<N> &dimsa,
		const permutation<N> &perma, const double &c, const double &d) {

	permutation<N> perma_i(perma, true);
	dimensions<N> dimsb(dimsa);
	dimsb.permute(perma);

	sequence<N, size_t>  map(0);
		for (size_t i = 0; i < N ; i++) {
		map[i] = i;
	}
	perma_i.apply(map);

	std::cout << "\n Map: " << map[0] << ", "<< map[1] << ", "<< map[2] << ", " << map[3] <<"\n";

	//chose driver according to tensor dimensionality N
	switch(N)	{
	case 2:
	{
		//get b increments using the map
		uint2 b_incrs = make_uint2(dimsb.get_increment(map[1]), dimsb.get_increment(map[0]) );
		std::cout << "dimsb dimensions: " << dimsb.get_dim(0) << ", " << dimsb.get_dim(1) << "\n";
		std::cout << "b_incrs: " << b_incrs.x << ", " << b_incrs.y << "\n " ;
		// setup execution parameters
		dim3 threads(dimsa.get_dim(1));
		dim3 grid(dimsa.get_dim(0));

		return new cuda_kern_copy_2d(pa, pb, threads, grid, b_incrs, c, d);
//		return NULL;
	}
	case 4:
	{
		//get b increments using the map
		uint4 b_incrs = make_uint4(dimsb.get_increment(map[3]), dimsb.get_increment(map[2]), dimsb.get_increment(map[1]), dimsb.get_increment(map[0]) );
		std::cout << "dimsb dimensions: " << dimsb.get_dim(0) << ", " << dimsb.get_dim(1) << ", " << dimsb.get_dim(2)  << ", " << dimsb.get_dim(3) << "\n";

		std::cout << "b_incrs: " << b_incrs.x << ", " << b_incrs.y << ", " << b_incrs.z << ", " << b_incrs.w << "\n " ;
		// setup execution parameters
		dim3 threads(dimsa.get_dim(3), dimsa.get_dim(2));
		dim3 grid(dimsa.get_dim(1), dimsa.get_dim(0));

		return new cuda_kern_copy_4d(pa, pb, threads, grid, b_incrs, c, d);
	}
	case 6:
	{
		//get incriments using the map
		uint3 b_incrs1 = make_uint3(dimsb.get_increment(map[5]), dimsb.get_increment(map[4]), dimsb.get_increment(map[3]));
		uint3 b_incrs2 = make_uint3(dimsb.get_increment(map[2]), dimsb.get_increment(map[1]), dimsb.get_increment(map[0]));
//
//		// setup execution parameters
		dim3 threads(dimsa.get_dim(5), dimsa.get_dim(4), dimsa.get_dim(3));
		dim3 grid(dimsa.get_dim(2), dimsa.get_dim(1), dimsa.get_dim(0));

		return new cuda_kern_copy_6d(pa, pb, threads, grid, b_incrs1, b_incrs2, c, d);
	}
	//no default copy
	default:
//		return new cuda_kern_copy_generic<N>(pa, pb, c);
		return NULL;
	}

//	cuda_kern_copy_generic<N> cuda_kern_generic(pa, pb, dimsa, c);
//	cuda_add_kern.m_dimsa = dimsa;
//	cuda_add_kern.m_c = c;

//	return this;

//	return new cuda_kern_copy_generic(cuda_kern_generic);
}

template cuda_kern_copy_generic *cuda_kern_copy_generic::match(const double *pa, double *pb, const dimensions<2> &dimsa,
		const permutation<2> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(const double *pa, double *pb, const dimensions<4> &dimsa,
		const permutation<4> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(const double *pa, double *pb, const dimensions<6> &dimsa,
		const permutation<6> &perma, const double &c, const double &d);

} // namespace libtensor
