#include <libvmm/std_allocator.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_import_raw.h>
#include <libtensor/btod/btod_tridiagonalize.h>
#include <libtensor/core/block_tensor.h>
#include "btod_tridiagonalize_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_tridiagonalize_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();

}


/**	\tridiagonalize matrix 3x3
 **/
void btod_tridiagonalize_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "btod_tridiagonalize_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {
		//initial symmetric matrix
		double matrix[9] = { 1, 3, 4, 3, 2, 8, 4, 8, 3};

		index<2> i1a, i1b;
		i1b[0] = 2; i1b[1] = 2;

		index<1> i2a, i2b;
		i2b[0] = 2;

		dimensions<2> dims1(index_range<2>(i1a, i1b));
		dimensions<1> dims2(index_range<1>(i2a, i2b));

		block_index_space<2> bis1(dims1);
		block_index_space<1> bis2(dims2);

		block_tensor<2, double, allocator_t> bta(bis1);//input matrix
		block_tensor<2, double, allocator_t> btb(bis1);//output matrix
		block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
													// symmetric ->tridiagonal
		block_tensor<2, double, allocator_t> btb_ref(bis1);//output matrix

		btod_import_raw<2>(matrix, dims1).perform(bta);

		//tridiagonalization
		btod_tridiagonalize(bta).perform(btb,S);

	//	Prepare the reference
		double matrixreference[9] = {1,-5,0,-5,10.32,1.76,0,1.76,-5.32};
		btod_import_raw<2>(matrixreference, dims1).perform(btb_ref);

	//	Compare against the reference
		compare_ref<2>::compare(testname, btb, btb_ref, 1e-5);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\tridiagonalize matrix 4x4 with fragmentation
 **/
void btod_tridiagonalize_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "btod_tridiagonalize_test::test_2()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

		double matrix[16] = { 4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2,1,-2,-1};
		//matrix with data

		index<2> i1a, i1b;
		i1b[0] = 3; i1b[1] = 3;

		index<1> i2a, i2b;
		i2b[0] = 3;

		dimensions<2> dims1(index_range<2>(i1a, i1b));
		dimensions<1> dims2(index_range<1>(i2a, i2b));

		block_index_space<2> bis1(dims1);
		block_index_space<1> bis2(dims2);

		mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;

		bis1.split(splmsk2, 2);

		block_tensor<2, double, allocator_t> bta(bis1);//input matrix
		block_tensor<2, double, allocator_t> btb(bis1);//output matrix
		block_tensor<2, double, allocator_t> btb_ref(bis1);
		block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
													// symmetric ->tridiagonal

		btod_import_raw<2>(matrix, dims1).perform(bta);

		//tridiagonalization
		btod_tridiagonalize(bta).perform(btb,S);

	//	Prepare the reference
		double matrixreference[16] = {4,-3,0,0,-3,3.33333,-1.66667,0,0,
				-1.66667,-1.32,0.906667,0,0,0.906667,1.98667};
		btod_import_raw<2>(matrixreference, dims1).perform(btb_ref);

	//	Compare against the reference
		compare_ref<2>::compare(testname, btb, btb_ref, 1e-5);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

/**	\tridiagonalize matrix 5x5 with fragmentation
 **/
void btod_tridiagonalize_test::test_3() throw(libtest::test_exception) {

	static const char *testname = "btod_tridiagonalize_test::test_3()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {
		double matrix[25] = { 2, -1,-1,0,0,-1,3,0,-2,0,-1,0,4,2,1,0,-2,2,8,3,0,
				0,1,3,9};
			//matrix with data

			index<2> i1a, i1b;
			i1b[0] = 4; i1b[1] = 4;

			index<1> i2a, i2b;
			i2b[0] = 4;

			dimensions<2> dims1(index_range<2>(i1a, i1b));
			dimensions<1> dims2(index_range<1>(i2a, i2b));

			block_index_space<2> bis1(dims1);
			block_index_space<1> bis2(dims2);

			//splitting
			mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
			mask<1> splmsk3; splmsk3[0] = true;

			bis1.split(splmsk2, 3);
			bis2.split(splmsk3, 3);

			block_tensor<2, double, allocator_t> bta(bis1);//input matrix
			block_tensor<2, double, allocator_t> btb(bis1);//output matrix
			block_tensor<2, double, allocator_t> btb_ref(bis1);//output matrix
			block_tensor<2, double, allocator_t> S(bis1);//matrix of
			//transformation symmetric ->tridiagonal

			btod_import_raw<2>(matrix, dims1).perform(bta);

			//tridiagonalization
			btod_tridiagonalize(bta).perform(btb,S);

	//	Prepare the reference
			double matrixreference[25] = {2,1.41421,0,0,0,1.41421,3.5,0.866025,
				0,0,0, 0.866025, 7.83333, 4.71405,0,0, 0, 4.71405,6.66667,
				1.73205,0,0,0, 1.73205 ,6 };
			btod_import_raw<2>(matrixreference, dims1).perform(btb_ref);

	//	Compare against the reference
			compare_ref<2>::compare(testname, btb, btb_ref, 1e-5);

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
