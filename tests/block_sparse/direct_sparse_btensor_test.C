#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include "direct_sparse_btensor_test.h"

namespace libtensor {

void direct_sparse_btensor_test::perform() throw(libtest::test_exception) {

    test_subtract_then_contract();
}

void direct_sparse_btensor_test::test_subtract_then_contract() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_subtract_then_contract()";

    //Row major
    double A_arr[20] = {1,2,3,4,5,
                        6,7,8,9,10,
                        11,12,13,14,15,
                        16,17,18,19,29};

    //Row major
    double B_arr[20] = {21,26,31,36,41,
                        22,27,32,37,42,
                        23,28,33,38,43,
                        24,29,34,39,44};


    //Row major
    double C_correct_arr[20] = {20,24,28,32,36,
                                16,20,24,28,32,
                                12,16,20,24,28,
                                8,12,16,20,15};

    sparse_bispace<1> spb_i(4);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(2);
    spb_i.split(split_points_i);

    sparse_bispace<1> spb_j(5);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);


    /*sparse_bispace<2> spb_A = spb_i | spb_j;*/
    /*sparse_btensor<2> A(spb_A,A_arr);*/
    /*sparse_btensor<2> B(spb_A,B_arr);*/
    /*direct_sparse_btensor<2> C(spb_A);*/

    /*letter i,j; */
    /*C(i|j) = B(i|j) - A(i|j);*/
    /*C*/

    /*direct_sparse_btensor */
}

}
