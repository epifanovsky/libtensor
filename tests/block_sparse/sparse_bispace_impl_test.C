#include "sparse_bispace_impl_test.h"
#include <libtensor/block_sparse/sparse_bispace_impl.h>

using namespace std;

namespace libtensor {

void sparse_bispace_impl_test::perform() throw(libtest::test_exception)
{
    test_equality_2d();
    test_equality_2d_sparse();

    /*test_permute_2d_10();*/
    /*test_permute_3d_dense_sparse_021();*/
    /*test_permute_3d_non_contiguous_sparsity();*/
    /*test_permute_3d_fully_sparse_210();*/
}

/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_impl_test::test_equality_2d() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_equality_2d";

    size_t sp_0[2] = {1,3};  
    subspace sub_0(5,idx_list(sp_0,sp_0+2));

    size_t sp_1[2] = {2,5};  
    subspace sub_1(6,idx_list(sp_1,sp_1+2));

    sparse_bispace_impl spb_0(sub_0,sub_1);
    sparse_bispace_impl spb_1(sub_0,sub_1);

    if(!(spb_0 == spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned false");
    }
}

void sparse_bispace_impl_test::test_equality_2d_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_equality_2d_sparse()";

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    size_t sp_0[3] = {2,5,9};
    vector<subspace> subspaces(2,subspace(11,idx_list(sp_0,sp_0+3)));

    //Specify different sets of significant blocks
    std::vector< sequence<2,size_t> > sig_blocks_0(4);
    sig_blocks_0[0][0] = 0; 
    sig_blocks_0[0][1] = 1;
    sig_blocks_0[1][0] = 1;
    sig_blocks_0[1][1] = 2;
    sig_blocks_0[2][0] = 2;
    sig_blocks_0[2][1] = 3;
    sig_blocks_0[3][0] = 3;
    sig_blocks_0[3][1] = 2;

    std::vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 1;//! Changed this one value
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    sparse_bispace_impl two_d_0(subspaces,sparse_block_tree(sig_blocks_0,subspaces));
    sparse_bispace_impl two_d_1(subspaces,sparse_block_tree(sig_blocks_0,subspaces));
    sparse_bispace_impl two_d_2(subspaces,sparse_block_tree(sig_blocks_1,subspaces));

    if(two_d_0 != two_d_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }

    if(two_d_0 == two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::operator==(...) returned incorrect value");
    }
}

#if 0
void sparse_bispace_impl_test::test_permute_2d_10() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_impl_test::test_permute_2d_10()";

    size_t sp_arr_0 = {1,3};
    subspace sub_0(5,idx_list(sp_arr_0,sp_arr_0+2));
    size_t sp_arr_1 = {2,5};
    subspace sub_1(5,idx_list(sp_arr_1,sp_arr_1+2));

    sparse_bispace_impl two_d(subspaces)
    swap(subspaces.begin(),subspaces.begin()+1);
    sparse_bispace_impl correct(subspaces)

    runtime_permutation perm;
    perm.permute(0,1);
    if(two_d.permute(perm) != (spb_2 | spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace_impl::permute(...) returned incorrect value");

    }
}
#endif

} // namespace libtensor
