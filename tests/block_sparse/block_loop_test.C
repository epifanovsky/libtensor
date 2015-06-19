#include "block_loop_test.h" 
#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/sparse_bispace_impl.h>
#include <libtensor/block_sparse/block_kernel_contract2.h>

using namespace std; 
namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_contract2();
    test_contract2_2d_2d_sparse_dense();
    /*test_contract2_3d_2d_sparse_sparse();*/
}

void block_loop_test::test_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_contract2()";

    size_t sp_i[3] = {2,5,9};
    subspace sub_i(11,idx_list(sp_i,sp_i+3));

    size_t sp_j[2] = {2,5};
    subspace sub_j(9,idx_list(sp_j,sp_j+2));

    size_t sp_k[3] = {1,4,8};
    subspace sub_k(10,idx_list(sp_k,sp_k+3));


    //Cij = Aik Bkj

    vector<block_loop> loops;
    idx_pair_list i_t_igs(1,idx_pair(0,0));
    i_t_igs.push_back(idx_pair(1,0));
    loops.push_back(block_loop(sub_i,i_t_igs,i_t_igs));

    idx_pair_list j_t_igs(1,idx_pair(0,1));
    j_t_igs.push_back(idx_pair(2,1));
    loops.push_back(block_loop(sub_j,j_t_igs,j_t_igs));

    idx_pair_list k_t_igs(1,idx_pair(1,1));
    k_t_igs.push_back(idx_pair(2,0));
    loops.push_back(block_loop(sub_k,k_t_igs,k_t_igs));

    vector<idx_list> orig_ig_offs(3,idx_list(2));
    orig_ig_offs[0][0] = 9;
    orig_ig_offs[0][1] = 1;
    orig_ig_offs[1][0] = 10;
    orig_ig_offs[1][1] = 1;
    orig_ig_offs[2][0] = 9;
    orig_ig_offs[2][1] = 1;



    vector<idx_list> ig_offs(orig_ig_offs);
    vector<idx_list> block_dims(3,idx_list(2,0));
    loops[0].apply_offsets(ig_offs);
    loops[0].apply_dims(block_dims);
    loops[1].apply_offsets(ig_offs);
    loops[1].apply_dims(block_dims);
    loops[2].apply_offsets(ig_offs);
    loops[2].apply_dims(block_dims);

    vector<idx_list> c_ig_offs(3,idx_list(2,0));
    vector<idx_list> c_block_dims(3,idx_list(2,2));
    c_block_dims[1][1] = 1; 
    c_block_dims[2][0] = 1; 

    if(ig_offs != c_ig_offs)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_offsets() returned incorrect ig_offs");

    if(block_dims != c_block_dims)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_dims() returned incorrect block_dims");

    ig_offs = orig_ig_offs;
    block_dims = vector<idx_list>(3,idx_list(2,0));
    ++loops[2];
    loops[0].apply_offsets(ig_offs);
    loops[0].apply_dims(block_dims);
    loops[1].apply_offsets(ig_offs);
    loops[1].apply_dims(block_dims);
    loops[2].apply_offsets(ig_offs);
    loops[2].apply_dims(block_dims);

    c_ig_offs[0][0] = 0;
    c_ig_offs[0][1] = 0;
    c_ig_offs[1][0] = 0;
    c_ig_offs[1][1] = 2;
    c_ig_offs[2][0] = 9;
    c_ig_offs[2][1] = 0;

    c_block_dims[1][1] = 3;
    c_block_dims[2][0] = 3;

    if(ig_offs != c_ig_offs)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_offsets() returned incorrect ig_offs");

    if(block_dims != c_block_dims)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_dims() returned incorrect block_dims");
}

//Cij = A(ik) Bjk
void block_loop_test::test_contract2_2d_2d_sparse_dense() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_contract2_2d_2d_sparse_dense()";

    size_t sp_i[3] = {2,5,9};
    subspace sub_i(11,idx_list(sp_i,sp_i+3));

    size_t sp_j[2] = {2,5};
    subspace sub_j(9,idx_list(sp_j,sp_j+2));

    size_t sp_k[3] = {1,4,8};
    subspace sub_k(10,idx_list(sp_k,sp_k+3));


    size_t keys_arr[3][2] = {{1,0},
                             {1,2},
                             {2,1}};
    size_t vals_arr[3] = {0,3,15};
    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 3; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+2));
    sparsity_data sd(2,keys);
    for(sparsity_data::iterator it = sd.begin(); it != sd.end(); ++it)
        it->second = idx_list(1,vals_arr[distance(sd.begin(),it)]);

    //Sparse loop over i
    vector<block_loop> loops;
    idx_pair_list i_t_igs(1,idx_pair(0,0));
    i_t_igs.push_back(idx_pair(1,0));
    idx_pair_list i_t_s(1,idx_pair(0,0));
    i_t_s.push_back(idx_pair(1,0));
    vector<bool> i_set_ig_off(2,true);
    i_set_ig_off[1] = false;
    loops.push_back(block_loop(sub_i,i_t_igs,i_t_s,i_set_ig_off,sd,0,idx_pair_list()));

    //Dense loop over j
    idx_pair_list j_t_igs(1,idx_pair(0,1));
    j_t_igs.push_back(idx_pair(2,0));
    loops.push_back(block_loop(sub_j,j_t_igs,j_t_igs));

    //Sparse loop over k
    idx_pair_list k_t_igs(1,idx_pair(1,0));
    k_t_igs.push_back(idx_pair(2,1));
    idx_pair_list k_t_s(1,idx_pair(1,1));
    k_t_s.push_back(idx_pair(2,1));
    vector<bool> k_set_ig_off(2,true);
    loops.push_back(block_loop(sub_k,k_t_igs,k_t_s,k_set_ig_off,sd,1,idx_pair_list(1,idx_pair(0,0))));

    loops[0].set_dependent_loop(loops[2]);


    //Starts off initialized to inner size of all index groups
    vector<idx_list> orig_ig_offs(3,idx_list(2));
    orig_ig_offs[1].resize(1);
    orig_ig_offs[0][0] = 9;
    orig_ig_offs[0][1] = 1;
    orig_ig_offs[1][0] = 1;
    orig_ig_offs[2][0] = 9;
    orig_ig_offs[2][1] = 1;
    vector<idx_list> ig_offs(orig_ig_offs);
    vector<idx_list> block_dims(3,idx_list(2,0));

    /*** i loop iteration 0, k loop iteration 0 ***/
    loops[0].apply_offsets(ig_offs);
    loops[0].apply_dims(block_dims);
    loops[1].apply_offsets(ig_offs);
    loops[1].apply_dims(block_dims);
    loops[2].apply_offsets(ig_offs);
    loops[2].apply_dims(block_dims);

    vector<idx_list> c_ig_offs(3,idx_list(2,0));
    c_ig_offs[1].resize(1);
    c_ig_offs[0][0] = 18;

    vector<idx_list> c_block_dims(3,idx_list(2,2));
    c_block_dims[0][0] = 3; 
    c_block_dims[1][0] = 3; 
    c_block_dims[1][1] = 1; 
    c_block_dims[2][1] = 1; 


    if(ig_offs != c_ig_offs)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_offsets() returned incorrect ig_offs");

    if(block_dims != c_block_dims)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_dims() returned incorrect block_dims");

    /*** i loop iteration 0, k loop iteration 1 ***/
    ++loops[2];
    ig_offs = orig_ig_offs;
    block_dims = vector<idx_list>(3,idx_list(2,0));

    loops[0].apply_offsets(ig_offs);
    loops[0].apply_dims(block_dims);
    loops[1].apply_offsets(ig_offs);
    loops[1].apply_dims(block_dims);
    loops[2].apply_offsets(ig_offs);
    loops[2].apply_dims(block_dims);

    c_ig_offs[1][0] = 3;
    c_ig_offs[2][1] = 8;

    c_block_dims[0][0] = 3; 
    c_block_dims[1][0] = 3; 
    c_block_dims[1][1] = 4; 
    c_block_dims[2][1] = 4; 

    if(ig_offs != c_ig_offs)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_offsets() returned incorrect ig_offs");

    if(block_dims != c_block_dims)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_dims() returned incorrect block_dims");


    ++loops[2];
    if(!loops[2].done())
        fail_test(test_name,__FILE__,__LINE__,
          "k loop done() did not return true");

    /*** i loop iteration 1, k loop iteration 0 ***/
    ig_offs = orig_ig_offs;
    block_dims = vector<idx_list>(3,idx_list(2,0));
    ++loops[0];
    loops[0].apply_offsets(ig_offs);
    loops[0].apply_dims(block_dims);
    loops[1].apply_offsets(ig_offs);
    loops[1].apply_dims(block_dims);
    loops[2].apply_offsets(ig_offs);
    loops[2].apply_dims(block_dims);

    c_ig_offs = vector<idx_list>(3,idx_list(2,0));
    c_ig_offs[1].resize(1);
    c_ig_offs[0][0] = 45;
    c_ig_offs[1][0] = 15;
    c_ig_offs[2][1] = 2;

    c_block_dims[0][0] = 4; 
    c_block_dims[1][0] = 4; 
    c_block_dims[1][1] = 3; 
    c_block_dims[2][1] = 3; 

    if(ig_offs != c_ig_offs)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_offsets() returned incorrect ig_offs");

    if(block_dims != c_block_dims)
        fail_test(test_name,__FILE__,__LINE__,
          "block_loop::apply_dims() returned incorrect block_dims");

    if(loops[2].done())
        fail_test(test_name,__FILE__,__LINE__,
          "k loop done() did not return false");

    /*** both sparse loops now should be done after incr of k ***/
    ++loops[2];
    if(!loops[2].done())
        fail_test(test_name,__FILE__,__LINE__,
          "k loop done() did not return true");
    ++loops[0];
    if(!loops[0].done())
        fail_test(test_name,__FILE__,__LINE__,
          "i loop done() did not return true");
    if(!loops[2].done())
        fail_test(test_name,__FILE__,__LINE__,
          "k loop done() did not return true");
}

void block_loop_test::test_contract2_3d_2d_sparse_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_contract2_3d_2d_sparse_sparse()";

    //Multiplication of the form A*B, no transposes
    //dimensions: i = 3, j = 4, k = 5, l = 6

    //Block major     
    double A_arr[26] = { //i = 0 j = 0 k = 1
                         1,
                         2,

                         //i = 0 j = 1 k = 0
                         3,4,
                         5,6,

                         //i = 0 j = 1 k = 2
                         7,8,
                         9,10,

                         //i = 1 j = 0 k = 1
                         11,
                         12,
                         13,
                         14,

                         //i = 1 j = 1 k = 1
                         15,
                         16,
                         17,
                         18,

                         //i = 1 j = 1 k = 2
                         19,20,
                         21,22,
                         23,24,
                         25,26};

                       
    //Block major
    double B_arr[19] = { //k = 0 l = 1
                         1,2,3,
                         4,5,6,

                         //k = 0 l = 2
                         7,
                         8,

                         //k = 1 l = 2
                         9,

                         //k = 2 l = 0
                         10,11,
                         12,13,

                         //k = 2 l = 1
                         14,15,16,
                         17,18,19};

    //Block major
    double C_correct_arr[42] = {//i = 0 j = 0 l = 2
                                9,
                                18,

                                //i = 0 j = 1 l = 0
                                166,181,
                                210,229,
                                
                                //i = 0 j = 1 l = 1
                                253,275,297,
                                325,355,385,
                                
                                //i = 0 j = 1 l = 2
                                53,
                                83,

                                //i = 1 j = 0 l = 2 
                                99,
                                108,
                                117,
                                126,

                                //i = 1 j = 1 l = 0
                                430,469, 
                                474,517,
                                518,565,
                                562,613,

                                //i = 1 j = 1 l = 1
                                606,645,684,
                                668,711,754,
                                730,777,824,
                                792,843,894,
    
                                //i = 1 j = 1 l = 2
                                135,
                                144,
                                153,
                                162};
    double C_arr[42] = {0};

    size_t sp_i[1] = {1};
    subspace sub_i(3,idx_list(sp_i,sp_i+1));

    size_t sp_j[1] = {2};
    subspace sub_j(4,idx_list(sp_j,sp_j+1));

    size_t sp_k[2] = {2,3};
    subspace sub_k(5,idx_list(sp_k,sp_k+2));

    size_t sp_l[2] = {2,5};
    subspace sub_l(6,idx_list(sp_l,sp_l+2));

    size_t keys_arr_A[6][3] = {{0,0,1},
                               {0,1,0},
                               {0,1,2},
                               {1,0,1},
                               {1,1,1},
                               {1,1,2}};
    size_t vals_arr_A[6] = {0,2,6,10,14,18};
    vector<idx_list> keys_A;
    for(size_t key_idx = 0; key_idx < 6; ++key_idx)
        keys_A.push_back(idx_list(keys_arr_A[key_idx],keys_arr_A[key_idx]+3));
    sparsity_data sd_A(3,keys_A);
    for(sparsity_data::iterator it = sd_A.begin(); it != sd_A.end(); ++it)
        it->second = idx_list(1,vals_arr_A[distance(sd_A.begin(),it)]);

    size_t keys_arr_B[5][2] = {{0,1},
                               {0,2},
                               {1,2},
                               {2,0},
                               {2,1}};

    size_t vals_arr_B[5] = {0,6,8,9,13};
    vector<idx_list> keys_B;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        keys_B.push_back(idx_list(keys_arr_B[key_idx],keys_arr_B[key_idx]+2));
    sparsity_data sd_B(2,keys_B);
    for(sparsity_data::iterator it = sd_B.begin(); it != sd_B.end(); ++it)
        it->second = idx_list(1,vals_arr_B[distance(sd_B.begin(),it)]);

    size_t keys_arr_C[8][3] = {{0,0,2},
                               {0,1,0},
                               {0,1,1},
                               {0,1,2},
                               {1,0,2},
                               {1,1,0},
                               {1,1,1},
                               {1,1,2}};
    size_t vals_arr_C[8] = {0,2,6,12,14,18,26,38};
    vector<idx_list> keys_C;
    for(size_t key_idx = 0; key_idx < 8; ++key_idx)
        keys_C.push_back(idx_list(keys_arr_C[key_idx],keys_arr_C[key_idx]+3));
    sparsity_data sd_C(3,keys_C);
    for(sparsity_data::iterator it = sd_C.begin(); it != sd_C.end(); ++it)
        it->second = idx_list(1,vals_arr_C[distance(sd_C.begin(),it)]);

    vector<subspace> subspaces(1,sub_i);
    subspaces.push_back(sub_j);
    subspaces.push_back(sub_l);
    sparse_bispace_impl spb_C(subspaces,vector<sparsity_data>(1,sd_C),idx_list(1,0));
    subspaces[2] = sub_k;
    sparse_bispace_impl spb_A(subspaces,vector<sparsity_data>(1,sd_A),idx_list(1,0));
    subspaces = vector<subspace>(2,sub_k);
    subspaces[1] = sub_l;
    sparse_bispace_impl spb_B(subspaces,vector<sparsity_data>(1,sd_B),idx_list(1,0));
    vector<sparse_bispace_impl> bispaces(1,spb_C);
    bispaces.push_back(spb_A);
    bispaces.push_back(spb_B);

    //Cijl = Aijk Bkl
    idx_list C_fuse_inds_A(1,0);
    C_fuse_inds_A.push_back(1);
    idx_list C_fuse_inds_B(1,2);
    C_fuse_inds_B.push_back(3);
    idx_list C_fuse_inds_B_perm(1,1);
    C_fuse_inds_B_perm.push_back(0);
    sparsity_data sd_fused = sd_C.fuse(sd_A,C_fuse_inds_A,C_fuse_inds_A).fuse(sd_B,C_fuse_inds_B,C_fuse_inds_B_perm);

    //Sparse loop over i
    vector<block_loop> loops;
    idx_pair_list i_t_igs(1,idx_pair(0,0));
    i_t_igs.push_back(idx_pair(1,0));
    idx_pair_list i_t_s(1,idx_pair(0,0));
    i_t_s.push_back(idx_pair(1,0));
    vector<bool> i_set_ig_off(2,false);
    loops.push_back(block_loop(sub_i,i_t_igs,i_t_s,i_set_ig_off,sd_fused,0,idx_pair_list()));

    //Sparse loop over j
    idx_pair_list j_t_igs(1,idx_pair(0,0));
    j_t_igs.push_back(idx_pair(1,0));
    idx_pair_list j_t_s(1,idx_pair(0,1));
    j_t_s.push_back(idx_pair(1,1));
    vector<bool> j_set_ig_off(2,false);
    loops.push_back(block_loop(sub_j,j_t_igs,j_t_s,j_set_ig_off,sd_fused,1,idx_pair_list()));

    //Sparse loop over l
    idx_pair_list l_t_igs(1,idx_pair(0,0));
    l_t_igs.push_back(idx_pair(2,0));
    idx_pair_list l_t_s(1,idx_pair(0,2));
    l_t_s.push_back(idx_pair(2,1));
    vector<bool> l_set_ig_off(2,true);
    idx_pair_list sd_off_map(1,idx_pair(0,0));
    sd_off_map.push_back(idx_pair(2,1));
    loops.push_back(block_loop(sub_l,l_t_igs,l_t_s,l_set_ig_off,sd_fused,2,sd_off_map));

    //Sparse loop over k
    idx_pair_list k_t_igs(1,idx_pair(1,0));
    k_t_igs.push_back(idx_pair(2,0));
    idx_pair_list k_t_s(1,idx_pair(1,2));
    k_t_s.push_back(idx_pair(2,0));
    vector<bool> k_set_ig_off(2,false);
    k_set_ig_off[0] = true;
    loops.push_back(block_loop(sub_k,k_t_igs,k_t_s,k_set_ig_off,sd_fused,3,idx_pair_list(1,idx_pair(1,0))));

    //Fully fused
    loops[0].set_dependent_loop(loops[1]);
    loops[1].set_dependent_loop(loops[2]);
    loops[2].set_dependent_loop(loops[3]);


    //We mimic sll here - hard-code pasting it kind of awful whatevz
    std::vector<double*> ptrs(1,C_arr);
    ptrs.push_back(A_arr);
    ptrs.push_back(B_arr);
    std::vector<double*> block_ptrs(ptrs.size());
    size_t c_loop_idx = 0;
    vector<idx_pair_list> ts_groups(4);
    ts_groups[0].push_back(idx_pair(0,0));
    ts_groups[0].push_back(idx_pair(1,0));
    ts_groups[1].push_back(idx_pair(0,1));
    ts_groups[1].push_back(idx_pair(1,1));
    ts_groups[2].push_back(idx_pair(0,2));
    ts_groups[2].push_back(idx_pair(2,1));
    ts_groups[3].push_back(idx_pair(1,2));
    ts_groups[3].push_back(idx_pair(2,0));
    block_kernel_contract2<double> bc2k(bispaces,ts_groups);

    vector<idx_list> ig_offs(3,idx_list(1,1));
    vector< vector<idx_list> > ig_off_grps(4,ig_offs);
    vector<idx_list> block_dims(3,idx_list(3,0));
    block_dims[2].pop_back();
    while(!(loops[0].done() && c_loop_idx == 0))
    {
        block_loop& c_loop = loops[c_loop_idx];
        if(c_loop_idx == 0)
            ig_off_grps[0] = ig_offs;
        else
            ig_off_grps[c_loop_idx] = ig_off_grps[c_loop_idx-1];

        if(!c_loop.done())
        {
            c_loop.apply_offsets(ig_off_grps[c_loop_idx]);
            c_loop.apply_dims(block_dims);

            if(c_loop_idx == loops.size() - 1)
            {
                for(size_t t_idx = 0; t_idx < ptrs.size(); ++t_idx)
                {
                    size_t offset = 0;
                    for(size_t ig = 0; ig < ig_offs[t_idx].size(); ++ig)
                    {
                        offset += ig_off_grps[c_loop_idx][t_idx][ig];
                    }
                    block_ptrs[t_idx] = ptrs[t_idx] + offset;
                }
                bc2k(block_ptrs,block_dims);
                ++c_loop;
            }
            else
                loops[++c_loop_idx].reset();
        }
        else
            ++loops[--c_loop_idx];
    }
}

} // namespace libtensor
