
#include <libtensor/block_sparse/sparse_btensor.h>
#include "sparse_btensor_test.h"

//TODO: REMOVE
#include <iostream>

namespace libtensor {
   
void sparse_btensor_test::perform() throw(libtest::test_exception) {

    test_get_bispace();

    /* Equality tests
     *
     */
    /*test_equality_true()*/

    /*test_str();*/
}

void sparse_btensor_test::test_get_bispace() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_get_bispace()";
    sparse_bispace<1> spb_1(5);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d = spb_1 | spb_2;
    sparse_btensor<2> sbt(two_d);
    if(sbt.get_bispace() != two_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::get_bispace(...) did not return two_d");
    }
}

void sparse_btensor_test::test_str() throw(libtest::test_exception)
{
    double mem_block_major[16] = { 1,2,5,6,
                                  3,4,7,8,
                                  9,10,13,14,
                                 11,12,15,16};


    sparse_bispace<1> N(4);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;
    sparse_btensor<2> bt(N2,mem_block_major);

    std::string correct_str("1 2\n5 6\n---\n3 4\n7 8\n---\n9 10\n13 14\n---\n11 12\n15 16\n");
    std::cout << bt.str();
    std::cout << "#########################\n"; 
    std::cout << correct_str << "\n";
}

#if 0
void sparse_btensor_test::test_equality_true() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_btensor_test::test_equality_true()";

    double mem_row_major[16] = { 1,2,3,4,
                                  5,6,7,8,
                                  9,10,11,12
                                 13,14,15,16};

    double mem_block_major[16] = { 1,2,5,6,
                                  3,4,7,8,
                                  9,10,13,14
                                 11,12,15,16};


    sparse_bispace<1> N(2);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    N.split(split_points);
    sparse_bispace<2> N2 = N|N;

    sparse_btensor<2> row_major(N2,mem_row_major);
    sparse_btensor<2> block_major(N2,mem_row_major);

    if(!(row_major == block_major))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor<N>::operator==(...) did not return true");
    }
}
#endif

} // namespace libtensor
