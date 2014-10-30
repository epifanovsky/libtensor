/* * sparse_loop_list_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparse_loop_grouper.h>
#include "sparse_loop_grouper_test.h"

using namespace std;

namespace libtensor {

void sparse_loop_grouper_test::perform() throw(libtest::test_exception) {
#if 0
    test_get_n_groups();
    test_get_bispaces_and_index_groups();
    test_get_offsets_and_sizes();
    test_get_bispaces_and_subspaces();
    test_get_block_dims();
    test_get_loops_for_groups();

    test_get_offsets_and_sizes_C_direct();
#endif
}

//Test fixtures
#if 0
namespace {

//This test fixture checks that get_bispaces_and_index_groups and 
//get_offsets_and_sizes correctly include information for the dense tensors coupled to the primary tree
//Cij = A(ki) Bkj
class dense_and_sparse_bispaces_test_f {
private:
    typedef pair<vector<block_loop>,
                 vector<sparse_bispace_any_order> > loops_bispaces_pair;

    static loops_bispaces_pair init_loops_and_bispaces()
    {
        vector<block_loop> loops;
        vector<sparse_bispace_any_order> bispaces;

        //Set up bispaces - need 6 blocks of size 2 each
        sparse_bispace<1> spb_i(12);
        vector<size_t> split_points_i;
        for(size_t i = 2; i < 12; i += 2)
        {
            split_points_i.push_back(i);
        }
        spb_i.split(split_points_i);
        sparse_bispace<1> spb_j = spb_i;
        sparse_bispace<1> spb_k = spb_i;

        //Sparsity for A
        size_t key_0_arr_A[2] = {2,1}; //offset 0
        size_t key_1_arr_A[2] = {3,2}; //offset 4
        size_t key_2_arr_A[2] = {4,5}; //offset 8
        size_t key_3_arr_A[2] = {5,4}; //offset 12

        std::vector< sequence<2,size_t> > sig_blocks_A(4);
        for(size_t i = 0; i < 2; ++i) sig_blocks_A[0][i] = key_0_arr_A[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_A[1][i] = key_1_arr_A[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_A[2][i] = key_2_arr_A[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_A[3][i] = key_3_arr_A[i];

        sparse_bispace<2> spb_C = spb_i | spb_j;
        sparse_bispace<2> spb_A = spb_k % spb_i << sig_blocks_A;
        sparse_bispace<2> spb_B = spb_k | spb_j;

        bispaces.push_back(spb_C);
        bispaces.push_back(spb_A);
        bispaces.push_back(spb_B);

        //Set up loop list
        loops.resize(3,block_loop(bispaces));
        //i loop
        loops[0].set_subspace_looped(0,0);
        loops[0].set_subspace_looped(1,1);

        //j loop
        loops[1].set_subspace_looped(0,1);
        loops[1].set_subspace_looped(2,1);

        //k loop
        loops[2].set_subspace_looped(1,0);
        loops[2].set_subspace_looped(2,0);

        return loops_bispaces_pair(loops,bispaces);
    }
public:
    vector<block_loop> loops;
    vector<sparse_bispace_any_order> bispaces;

    dense_and_sparse_bispaces_test_f() : loops(init_loops_and_bispaces().first), bispaces(init_loops_and_bispaces().second) {}
};

} // namespace unnamed
#endif

#if 0
void sparse_loop_grouper_test::test_get_n_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_n_groups()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    if(slg.get_n_groups() != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_n_groups(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_bispaces_and_index_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_bispaces_and_index_groups()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    vector<idx_pair_list> bispaces_and_index_groups = slg.get_bispaces_and_index_groups();

    vector<idx_pair_list> correct_bispaces_and_index_groups(2);
    correct_bispaces_and_index_groups[0].push_back(idx_pair(0,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(1,0));
    correct_bispaces_and_index_groups[0].push_back(idx_pair(2,0));

    correct_bispaces_and_index_groups[1].push_back(idx_pair(0,1));
    correct_bispaces_and_index_groups[1].push_back(idx_pair(2,1));

    if(bispaces_and_index_groups != correct_bispaces_and_index_groups)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_bispaces_and_index_groups(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_offsets_and_sizes() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_offsets_and_sizes()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    vector< vector<off_dim_pair_list> > correct_oas(2);
    off_dim_pair correct_0_0_arr[3] = {off_dim_pair(2,2),off_dim_pair(0,4),off_dim_pair(4,2)};
    off_dim_pair correct_0_1_arr[3] = {off_dim_pair(4,2),off_dim_pair(4,4),off_dim_pair(6,2)};
    off_dim_pair correct_0_2_arr[3] = {off_dim_pair(8,2),off_dim_pair(12,4),off_dim_pair(10,2)};
    off_dim_pair correct_0_3_arr[3] = {off_dim_pair(10,2),off_dim_pair(8,4),off_dim_pair(8,2)};

    correct_oas[0].push_back(off_dim_pair_list(correct_0_0_arr,correct_0_0_arr+3));
    correct_oas[0].push_back(off_dim_pair_list(correct_0_1_arr,correct_0_1_arr+3));
    correct_oas[0].push_back(off_dim_pair_list(correct_0_2_arr,correct_0_2_arr+3));
    correct_oas[0].push_back(off_dim_pair_list(correct_0_3_arr,correct_0_3_arr+3));

    off_dim_pair correct_1_0_arr[2] = {off_dim_pair(0,2),off_dim_pair(0,2)};
    off_dim_pair correct_1_1_arr[2] = {off_dim_pair(2,2),off_dim_pair(2,2)};
    off_dim_pair correct_1_2_arr[2] = {off_dim_pair(4,2),off_dim_pair(4,2)};
    off_dim_pair correct_1_3_arr[2] = {off_dim_pair(6,2),off_dim_pair(6,2)};
    off_dim_pair correct_1_4_arr[2] = {off_dim_pair(8,2),off_dim_pair(8,2)};
    off_dim_pair correct_1_5_arr[2] = {off_dim_pair(10,2),off_dim_pair(10,2)};

    correct_oas[1].push_back(off_dim_pair_list(correct_1_0_arr,correct_1_0_arr+2));
    correct_oas[1].push_back(off_dim_pair_list(correct_1_1_arr,correct_1_1_arr+2));
    correct_oas[1].push_back(off_dim_pair_list(correct_1_2_arr,correct_1_2_arr+2));
    correct_oas[1].push_back(off_dim_pair_list(correct_1_3_arr,correct_1_3_arr+2));
    correct_oas[1].push_back(off_dim_pair_list(correct_1_4_arr,correct_1_4_arr+2));
    correct_oas[1].push_back(off_dim_pair_list(correct_1_5_arr,correct_1_5_arr+2));

    if(slg.get_offsets_and_sizes()  != correct_oas)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_offsets_and_sizes(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_bispaces_and_subspaces() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_bispaces_and_subspaces()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    vector<idx_pair_list> bispaces_and_subspaces = slg.get_bispaces_and_subspaces();

    vector<idx_pair_list> correct_bispaces_and_subspaces(2);
    correct_bispaces_and_subspaces[0].push_back(idx_pair(0,0));
    correct_bispaces_and_subspaces[0].push_back(idx_pair(1,1));
    correct_bispaces_and_subspaces[0].push_back(idx_pair(1,0));
    correct_bispaces_and_subspaces[0].push_back(idx_pair(2,0));

    correct_bispaces_and_subspaces[1].push_back(idx_pair(0,1));
    correct_bispaces_and_subspaces[1].push_back(idx_pair(2,1));

    if(bispaces_and_subspaces != correct_bispaces_and_subspaces)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_bispaces_and_subspaces(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_block_dims() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_block_dims()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    vector<vector<dim_list> > block_dims = slg.get_block_dims();

    vector<vector<dim_list> > correct_block_dims(2);
    correct_block_dims[0].resize(4,dim_list(4,2));
    correct_block_dims[1].resize(6,dim_list(2,2));

    if(block_dims != correct_block_dims)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_block_dims(...) returned incorrect value");
    }
}

void sparse_loop_grouper_test::test_get_loops_for_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_loops_for_groups()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    sparse_loop_grouper slg(sparsity_fuser(tf.loops,tf.bispaces));

    vector<idx_list> loops_for_groups = slg.get_loops_for_groups();
    vector<idx_list> correct_loops_for_groups(2);
    //(ik)
    correct_loops_for_groups[0].push_back(0);
    correct_loops_for_groups[0].push_back(2);
    //j
    correct_loops_for_groups[1].push_back(1);


    if(loops_for_groups != correct_loops_for_groups)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_loops_for_groups(...) returned incorrect value");
    }
}

//This will show that the grouper comes up with the correct batch-relative offset for C when the batched index is dense
void sparse_loop_grouper_test::test_get_offsets_and_sizes_C_direct() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_grouper_test::test_get_offsets_and_sizes_C_direct()";

    dense_and_sparse_bispaces_test_f tf = dense_and_sparse_bispaces_test_f();
    idx_list direct_tensors(1,0);

    /*** FIRST BATCH ***/
    //Batch over i
    map<size_t,idx_pair> batches;
    batches[0] = idx_pair(0,5);
    sparse_loop_grouper slg_0(sparsity_fuser(tf.loops,tf.bispaces,direct_tensors,batches));

    //First loop group, first batch
    vector< vector<off_dim_pair_list> > correct_oas_0(2);
    off_dim_pair correct_0_0_0_arr[3] = {off_dim_pair(0,2),off_dim_pair(0,4),off_dim_pair(4,2)};
    off_dim_pair correct_0_0_1_arr[3] = {off_dim_pair(2,2),off_dim_pair(4,4),off_dim_pair(6,2)};
    off_dim_pair correct_0_0_2_arr[3] = {off_dim_pair(6,2),off_dim_pair(12,4),off_dim_pair(10,2)};

    correct_oas_0[0].push_back(off_dim_pair_list(correct_0_0_0_arr,correct_0_0_0_arr+3));
    correct_oas_0[0].push_back(off_dim_pair_list(correct_0_0_1_arr,correct_0_0_1_arr+3));
    correct_oas_0[0].push_back(off_dim_pair_list(correct_0_0_2_arr,correct_0_0_2_arr+3));

    //Second loop group, first and second batches (doesn't change)
    off_dim_pair correct_0_1_0_arr[2] = {off_dim_pair(0,2),off_dim_pair(0,2)};
    off_dim_pair correct_0_1_1_arr[2] = {off_dim_pair(2,2),off_dim_pair(2,2)};
    off_dim_pair correct_0_1_2_arr[2] = {off_dim_pair(4,2),off_dim_pair(4,2)};
    off_dim_pair correct_0_1_3_arr[2] = {off_dim_pair(6,2),off_dim_pair(6,2)};
    off_dim_pair correct_0_1_4_arr[2] = {off_dim_pair(8,2),off_dim_pair(8,2)};
    off_dim_pair correct_0_1_5_arr[2] = {off_dim_pair(10,2),off_dim_pair(10,2)};

    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_0_arr,correct_0_1_0_arr+2));
    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_1_arr,correct_0_1_1_arr+2));
    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_2_arr,correct_0_1_2_arr+2));
    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_3_arr,correct_0_1_3_arr+2));
    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_4_arr,correct_0_1_4_arr+2));
    correct_oas_0[1].push_back(off_dim_pair_list(correct_0_1_5_arr,correct_0_1_5_arr+2));

    if(slg_0.get_offsets_and_sizes()  != correct_oas_0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_offsets_and_sizes(...) returned incorrect value");
    }

    /*** SECOND BATCH ***/
    batches[0] = idx_pair(5,6);
    sparse_loop_grouper slg_1(sparsity_fuser(tf.loops,tf.bispaces,direct_tensors,batches));

    //First loop group, second batch
    vector< vector<off_dim_pair_list> > correct_oas_1(2);
    off_dim_pair correct_1_0_0_arr[3] = {off_dim_pair(0,2),off_dim_pair(8,4),off_dim_pair(8,2)};
    correct_oas_1[0].push_back(off_dim_pair_list(correct_1_0_0_arr,correct_1_0_0_arr+3));

    correct_oas_1[1] = correct_oas_0[1];

    if(slg_1.get_offsets_and_sizes()  != correct_oas_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_loop_grouper::get_offsets_and_sizes(...) returned incorrect value");
    }
}
#endif

} // namespace libtensor
