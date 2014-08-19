#include "sparse_bispace_impl_test.h"
#include <libtensor/block_sparse/sparse_bispace_impl.h>

using namespace std;

namespace libtensor {

void sparse_bispace_impl_test::perform() throw(libtest::test_exception)
{
    test_nd_equality_true();
    /*test_equality_false_sparsity_2d();*/
    /*test_equality_true_sparsity_2d();*/
}

/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_impl_test::test_nd_equality_true() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_nd_equality_true()";

    //First two
    vector<subspace> subspaces(1,subspace(5));
    vector<size_t> split_points_0; 
    split_points_0.push_back(1);
    split_points_0.push_back(3);
    subspaces[0].split(split_points_0);

    subspaces.push_back(subspace(6));
    vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    subspaces[1].split(split_points_1);

    sparse_bispace_impl spb_i_0(subspaces);
    sparse_bispace_impl spb_i_1(subspaces);

    if(!(spb_i_0 == spb_i_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned false");
    }
}

#if 0
void sparse_bispace_test::test_equality_false_sparsity_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_equality_false_sparsity_2d()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    //Specify different sets of significant blocks
    std::vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 3;
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    std::vector< sequence<2,size_t> > sig_blocks_2(4);
    sig_blocks_2[0][0] = 0; 
    sig_blocks_2[0][1] = 1;
    sig_blocks_2[1][0] = 1;
    sig_blocks_2[1][1] = 2;
    sig_blocks_2[2][0] = 2;
    sig_blocks_2[2][1] = 1;//! Changed this one value
    sig_blocks_2[3][0] = 3;
    sig_blocks_2[3][1] = 2;

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks_1; 
    sparse_bispace<2> two_d_2 = spb_1 % spb_1 << sig_blocks_2; 

    if(two_d_1 == two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator==(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_equality_true_sparsity_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_equality_true_sparsity_2d()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    //Specify different sets of significant blocks
    std::vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 3;
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks_1; 
    sparse_bispace<2> two_d_2 = spb_1 % spb_1 << sig_blocks_1; 

    if(two_d_1 != two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator==(...) returned incorrect value");
    }
}
#endif

} // namespace libtensor
