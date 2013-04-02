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
}


template<size_t N>
cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<N> &dimsa,
		const permutation<N> &perma, const double &c, const double &d) {

	permutation<N> perma_i(perma, true);
	dimensions<N> dimsb(dimsa);
	dimsb.permute(perma);

	sequence<N, size_t>  map(0);
		for (size_t i = 0; i < N ; i++) {
		map[i] = i;
	}
	perma_i.apply(map);

	//chose driver according to tensor dimensionality N
	switch(N)	{
	case 2:
	{
		const int THREADS_PER_BLOCK = 512;
//		const int THREADS_PER_BLOCK = 2;

		uint2 dims = make_uint2(dimsa.get_dim(0), dimsa.get_dim(1) );
		//get b increments using the map
		uint2 b_incrs = make_uint2(dimsb.get_increment(map[1]), dimsb.get_increment(map[0]) );
		// setup execution parameters
		dim3 threads;
		// setup execution parameters
		threads.x = (dimsa.get_dim(1) < THREADS_PER_BLOCK) ? dimsa.get_dim(1) : THREADS_PER_BLOCK;
		dim3 grid(dimsa.get_dim(0));

		return new cuda_kern_copy_2d(pa, pb, threads, grid, b_incrs, dims, c, d);
	}
	case 4:
	{
		uint4 dims = make_uint4(dimsa.get_dim(3), dimsa.get_dim(2), dimsa.get_dim(1), dimsa.get_dim(0) );
		//get b increments using the map
		uint4 b_incrs = make_uint4(dimsb.get_increment(map[3]), dimsb.get_increment(map[2]), dimsb.get_increment(map[1]), dimsb.get_increment(map[0]) );

		// setup execution parameters
//		dim3 threads(dimsa.get_dim(3), dimsa.get_dim(2));
		//maximum possible block configuration is 32x32
		dim3 threads;
		threads.x = (dimsa.get_dim(3) > 32) ? 32 : dimsa.get_dim(3);
		threads.y = (dimsa.get_dim(2) > 32) ? 32 : dimsa.get_dim(2);
		dim3 grid(dimsa.get_dim(1), dimsa.get_dim(0));

		return new cuda_kern_copy_4d(pa, pb, threads, grid, b_incrs, dims, c, d);
	}
	case 6:
	{
		uint3 dims2 = make_uint3(dimsa.get_dim(5), dimsa.get_dim(4), dimsa.get_dim(3) );
		//get incriments using the map
		uint3 b_incrs1 = make_uint3(dimsb.get_increment(map[5]), dimsb.get_increment(map[4]), dimsb.get_increment(map[3]));
		uint3 b_incrs2 = make_uint3(dimsb.get_increment(map[2]), dimsb.get_increment(map[1]), dimsb.get_increment(map[0]));
//
//		// setup execution parameters
		//maximum possible block configuration is 8x8x16
//		dim3 threads(dimsa.get_dim(5), dimsa.get_dim(4), dimsa.get_dim(3));
		dim3 threads;
		threads.x = (dimsa.get_dim(5) > 16) ? 16 : dimsa.get_dim(5);
		threads.y = (dimsa.get_dim(4) > 8) ? 8 : dimsa.get_dim(4);
		threads.z = (dimsa.get_dim(3) > 8) ? 8 : dimsa.get_dim(3);
//		threads.x = (dimsa.get_dim(5) > 2) ? 2 : dimsa.get_dim(5);
//		threads.y = (dimsa.get_dim(4) > 2) ? 2 : dimsa.get_dim(4);
//		threads.z = (dimsa.get_dim(3) > 2) ? 2 : dimsa.get_dim(3);
		dim3 grid(dimsa.get_dim(2), dimsa.get_dim(1), dimsa.get_dim(0));

		return new cuda_kern_copy_6d(pa, pb, threads, grid, b_incrs1, b_incrs2, dims2, c, d);
	}
	//no default copy
	default:
		return NULL;
	}

}

template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<1> &dimsa,
		const permutation<1> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<2> &dimsa,
		const permutation<2> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<3> &dimsa,
		const permutation<3> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<4> &dimsa,
		const permutation<4> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<5> &dimsa,
		const permutation<5> &perma, const double &c, const double &d);
template cuda_kern_copy_generic *cuda_kern_copy_generic::match(cuda_pointer<const double> pa, cuda_pointer<double> pb, const dimensions<6> &dimsa,
		const permutation<6> &perma, const double &c, const double &d);

} // namespace libtensor
