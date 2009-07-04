#include <libvmm.h>
#include <libtensor.h>
#include "block_index_space_test.h"

namespace libtensor {

void block_index_space_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
	test_3();

}

void block_index_space_test::test_1() throw(libtest::test_exception) {

	try {

	index<1> i_0;
	index<1> i_1; i_1[0] = 1;
	index<1> i_2; i_2[0] = 2;
	index<1> i_3; i_3[0] = 3;
	index<1> i_4; i_4[0] = 4;
	index<1> i_7; i_7[0] = 7;
	index<1> i_9; i_9[0] = 9;

	dimensions<1> d_1(index_range<1>(i_0, i_0));
	dimensions<1> d_2(index_range<1>(i_0, i_1));
	dimensions<1> d_3(index_range<1>(i_0, i_2));
	dimensions<1> d_5(index_range<1>(i_0, i_4));
	dimensions<1> d_8(index_range<1>(i_0, i_7));
	dimensions<1> d_10(index_range<1>(i_0, i_9));

	block_index_space<1> bis(d_10);

	if(!bis.get_dims().equals(d_10)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(1) Incorrect total dimensions");
	}
	if(!bis.get_block_index_dims().equals(d_1)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(1) Incorrect block index dimensions");
	}
	if(!bis.get_block_dims(i_0).equals(d_10)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(1) Incorrect block [0] dimensions");
	}

	bis.split(0, 2);

	if(!bis.get_dims().equals(d_10)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(2) Incorrect total dimensions");
	}
	if(!bis.get_block_index_dims().equals(d_2)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(2) Incorrect block index dimensions");
	}
	if(!bis.get_block_dims(i_0).equals(d_2)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(2) Incorrect block [0] dimensions");
	}
	if(!bis.get_block_dims(i_1).equals(d_8)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(2) Incorrect block [1] dimensions");
	}

	bis.split(0, 5);

	if(!bis.get_dims().equals(d_10)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(3) Incorrect total dimensions");
	}
	if(!bis.get_block_index_dims().equals(d_3)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(3) Incorrect block index dimensions");
	}
	if(!bis.get_block_dims(i_0).equals(d_2)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(3) Incorrect block [0] dimensions");
	}
	if(!bis.get_block_dims(i_1).equals(d_3)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(3) Incorrect block [1] dimensions");
	}
	if(!bis.get_block_dims(i_2).equals(d_5)) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, "(3) Incorrect block [2] dimensions");
	}

	} catch(exception &e) {
		fail_test("block_index_space_test::test_1()", __FILE__,
			__LINE__, e.what());
	}

}

void block_index_space_test::test_2() throw(libtest::test_exception) {

	try {

	index<1> i_0;
	index<1> i_1; i_1[0] = 1;
	index<1> i_2; i_2[0] = 2;
	index<1> i_3; i_3[0] = 3;

	dimensions<1> d_1(index_range<1>(i_0, i_0));
	dimensions<1> d_2(index_range<1>(i_0, i_1));
	dimensions<1> d_3(index_range<1>(i_0, i_2));

	block_index_space<1> bis(d_3);

	if(!bis.get_dims().equals(d_3)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(1) Incorrect total dimensions");
	}

	bis.split(0, 1);
	bis.split(0, 2);

	if(!bis.get_dims().equals(d_3)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(2) Incorrect total dimensions");
	}
	if(!bis.get_block_index_dims().equals(d_3)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(2) Incorrect block index dimensions");
	}
	if(!bis.get_block_dims(i_0).equals(d_1)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(2) Incorrect block [0] dimensions");
	}
	if(!bis.get_block_dims(i_1).equals(d_1)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(2) Incorrect block [1] dimensions");
	}
	if(!bis.get_block_dims(i_2).equals(d_1)) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, "(2) Incorrect block [2] dimensions");
	}

	} catch(exception &e) {
		fail_test("block_index_space_test::test_2()", __FILE__,
			__LINE__, e.what());
	}

}

void block_index_space_test::test_3() throw(libtest::test_exception) {

	try {

	index<2> i_00;
	index<2> i_01; i_01[1] = 1;
	index<2> i_02; i_02[1] = 2;
	index<2> i_03; i_03[1] = 3;
	index<2> i_10; i_10[0] = 1;
	index<2> i_11; i_11[0] = 1; i_11[1] = 1;
	index<2> i_12; i_12[0] = 1; i_12[1] = 2;
	index<2> i_13; i_13[0] = 1; i_13[1] = 3;
	index<2> i_20; i_20[0] = 2;
	index<2> i_21; i_21[0] = 2; i_21[1] = 1;
	index<2> i_22; i_22[0] = 2; i_22[1] = 2;
	index<2> i_23; i_23[0] = 2; i_23[1] = 3;
	index<2> i_30; i_30[0] = 3;
	index<2> i_31; i_31[0] = 3; i_31[1] = 1;
	index<2> i_32; i_32[0] = 3; i_32[1] = 2;
	index<2> i_33; i_33[0] = 3; i_33[1] = 3;
	index<2> i_55; i_55[0] = 5; i_55[1] = 5;

	dimensions<2> d_11(index_range<2>(i_00, i_00));
	dimensions<2> d_12(index_range<2>(i_00, i_01));
	dimensions<2> d_13(index_range<2>(i_00, i_02));
	dimensions<2> d_21(index_range<2>(i_00, i_10));
	dimensions<2> d_22(index_range<2>(i_00, i_11));
	dimensions<2> d_23(index_range<2>(i_00, i_12));
	dimensions<2> d_31(index_range<2>(i_00, i_20));
	dimensions<2> d_32(index_range<2>(i_00, i_21));
	dimensions<2> d_33(index_range<2>(i_00, i_22));
	dimensions<2> d_66(index_range<2>(i_00, i_55));

	block_index_space<2> bis(d_66);

	bis.split(0, 1);
	bis.split(0, 3);
	bis.split(1, 1);
	bis.split(1, 3);

	if(!bis.get_dims().equals(d_66)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect total dimensions");
	}
	if(!bis.get_block_index_dims().equals(d_33)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block index dimensions");
	}
	if(!bis.get_block_dims(i_00).equals(d_11)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [0,0] dimensions");
	}
	if(!bis.get_block_dims(i_01).equals(d_12)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [0,1] dimensions");
	}
	if(!bis.get_block_dims(i_02).equals(d_13)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [0,2] dimensions");
	}
	if(!bis.get_block_dims(i_10).equals(d_21)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [1,0] dimensions");
	}
	if(!bis.get_block_dims(i_11).equals(d_22)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [1,1] dimensions");
	}
	if(!bis.get_block_dims(i_12).equals(d_23)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [1,2] dimensions");
	}
	if(!bis.get_block_dims(i_20).equals(d_31)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [2,0] dimensions");
	}
	if(!bis.get_block_dims(i_21).equals(d_32)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [2,1] dimensions");
	}
	if(!bis.get_block_dims(i_22).equals(d_33)) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, "(1) Incorrect block [2,2] dimensions");
	}

	} catch(exception &e) {
		fail_test("block_index_space_test::test_3()", __FILE__,
			__LINE__, e.what());
	}

}


} // namespace libtensor
