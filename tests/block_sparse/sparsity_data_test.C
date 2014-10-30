#include <libtensor/block_sparse/sparsity_data.h>
#include <libtensor/block_sparse/range.h>
#include "sparsity_data_test.h"

using namespace std;

namespace libtensor {

//Test fixtures
namespace {

class two_d_test_f 
{
private:
    static size_t keys_arr[8][2];
public:
    sparsity_data sd;

    static vector<idx_list> init_keys() 
    {
        vector<idx_list> keys;
        for(size_t key_idx = 0; key_idx < 8; ++key_idx)
            keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+2));
        return keys;
    }

    two_d_test_f() : sd(2,init_keys())
    {
        size_t vals_arr[8] = {0,4,8,12,16,20,24,28};

        for(sparsity_data::iterator it = sd.begin(); it != sd.end(); ++it)
        {
            it->second.push_back(vals_arr[distance(sd.begin(),it)]);
        }
    }
};

size_t two_d_test_f::keys_arr[8][2] = {{1,2},
                                       {1,5},
                                       {2,3},
                                       {4,1},
                                       {4,4},
                                       {5,1},
                                       {5,2},
                                       {7,4}};

class three_d_test_f
{
private:
    static size_t keys_arr[21][3];
public:
    sparsity_data sd;

    static vector<idx_list> init_keys() 
    {
        vector<idx_list> keys;
        for(size_t key_idx = 0; key_idx < 21; ++key_idx)
            keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+3));
        return keys;
    }

    three_d_test_f() : sd(3,init_keys())
    {
        size_t vals_arr[21] = {0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160};

        for(sparsity_data::iterator it = sd.begin(); it != sd.end(); ++it)
        {
            it->second.push_back(vals_arr[distance(sd.begin(),it)]);
        }
    }
};

size_t three_d_test_f::keys_arr[21][3] = {{1,2,3},
                                          {1,2,7},
                                          {1,3,1},
                                          {1,5,9},
                                          {2,3,1},
                                          {2,4,2},
                                          {2,4,5},
                                          {2,6,3},
                                          {2,6,4},
                                          {4,1,4},
                                          {4,1,7},
                                          {4,2,2},
                                          {4,3,5},
                                          {4,3,6},
                                          {4,3,7},
                                          {5,1,4},
                                          {5,2,6},
                                          {5,2,7},
                                          {7,4,5},
                                          {7,4,6},
                                          {7,7,7}};

} // namespace unnamed

void sparsity_data_test::perform() throw(libtest::test_exception)
{
    test_zero_order();
    test_invalid_key_length();
    test_unsorted_keys();
    test_duplicate_keys();

    test_equality_2d();

    test_permute_3d();

    test_contract_3d_0();
    test_contract_3d_1();
    test_contract_3d_2();

    test_fuse_3d_2d();
    test_fuse_3d_3d_non_contiguous();
    test_fuse_3d_3d_multi_index();

    test_truncate_subspace_3d();
    test_insert_subspace_3d();
}

//Cannot have sparsity data of zero
void sparsity_data_test::test_zero_order() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_zero_order()";
    
    bool threw_exception = false;
    try
    {
        sparsity_data sd(0,vector<idx_list>(1,idx_list()));
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    
    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "Instantiating a zero-order sparsity_data object did not cause an exception");
    }
}

//All keys must be the same length
void sparsity_data_test::test_invalid_key_length() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_invalid_key_length()";
    
    //Put one too long in the middle
    bool threw_exception = false;
    vector<idx_list> keys(1,idx_list(2,1));
    keys.push_back(idx_list(3,1));
    keys.push_back(idx_list(2,1));
    try
    {
        sparsity_data sd(2,keys);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    
    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "Passing a key with invalid length did not throw exception");
    }
}

void sparsity_data_test::test_unsorted_keys() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_unsorted_keys()";

    //Middle one is out of order
    bool threw_exception = false;
    vector<idx_list> keys(1,idx_list(2,1));
    keys.push_back(idx_list(2,0));
    keys.push_back(idx_list(2,1));
    try
    {
        sparsity_data sd(2,keys);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    
    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "Passing unsorted keys did not throw exception");
    }
}

void sparsity_data_test::test_duplicate_keys() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_duplicate_keys()";

    //Middle one is out of order
    bool threw_exception = false;
    vector<idx_list> keys(1,idx_list(2,1));
    keys.push_back(idx_list(2,1));
    keys.push_back(idx_list(2,1));
    keys.push_back(idx_list(2,2));
    try
    {
        sparsity_data sd(2,keys);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    
    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "Passing duplicate keys did not throw exception");
    }
}


void sparsity_data_test::test_equality_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_equality_2d()";

    //True case
    size_t keys_arr[4][2] = {{0,0},
                             {0,1},
                             {1,1},
                             {1,2}};
    vector<idx_list> keys;
    for(size_t key_idx = 0; key_idx < 4; ++key_idx)
        keys.push_back(idx_list(keys_arr[key_idx],keys_arr[key_idx]+2));

    sparsity_data sd_0(2,keys);
    sparsity_data sd_1(2,keys);
    if(sd_0 != sd_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparsity_data::operator==(...) returned incorrect value");
    }

    //False case
    keys[1][1] = 5;
    sparsity_data sd_2(2,keys);
    
    if(sd_0 == sd_2 || sd_1 == sd_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                  "sparsity_data::operator==(...) returned incorrect value");
    }
}

void sparsity_data_test::test_permute_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_permute_3d()";

    three_d_test_f tf = three_d_test_f();

    //permute first and last index
    runtime_permutation perm(3);
    perm.permute(0,2);

    sparsity_data permuted_sd = tf.sd.permute(perm);

    //Benchmark result
    std::vector<idx_list> permuted_keys;

    size_t permuted_keys_arr[21][3] = {{1,3,1},
                                       {1,3,2},
                                       {2,2,4},
                                       {2,4,2},
                                       {3,2,1},
                                       {3,6,2},
                                       {4,1,4},
                                       {4,1,5},
                                       {4,6,2},
                                       {5,3,4},
                                       {5,4,2},
                                       {5,4,7},
                                       {6,2,5},
                                       {6,3,4},
                                       {6,4,7},
                                       {7,1,4},
                                       {7,2,1},
                                       {7,2,5},
                                       {7,3,4},
                                       {7,7,7},
                                       {9,5,1}};

    for(size_t key_idx = 0; key_idx < 21; ++key_idx)
        permuted_keys.push_back(idx_list(permuted_keys_arr[key_idx],permuted_keys_arr[key_idx]+3));

    size_t permuted_vals_arr[21] = {16,32,88,40,0,56,72,120,64,96,48,144,128,104,152,80,8,136,112,160,24};

    sparsity_data correct_sd(3,permuted_keys); 
    for(sparsity_data::iterator it = correct_sd.begin(); it != correct_sd.end(); ++it)
    {
        it->second.push_back(permuted_vals_arr[distance(correct_sd.begin(),it)]);
    }

    if(permuted_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "permute returned incorrect sparsity_data");
    }
}

void sparsity_data_test::test_contract_3d_0() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_contract_3d_0()";

    three_d_test_f tf = three_d_test_f();
    sparsity_data contr_sd = tf.sd.contract(0);

    //Build the correct tree
    size_t contr_keys_arr[17][2] = {{1,4},
                                    {1,7},
                                    {2,2},
                                    {2,3},
                                    {2,6},
                                    {2,7},
                                    {3,1},
                                    {3,5},
                                    {3,6},
                                    {3,7},
                                    {4,2},
                                    {4,5},
                                    {4,6},
                                    {5,9},
                                    {6,3},
                                    {6,4},
                                    {7,7}};

    vector<idx_list> contr_keys;
    for(size_t key_idx = 0; key_idx < 17; ++key_idx)
        contr_keys.push_back(idx_list(contr_keys_arr[key_idx],contr_keys_arr[key_idx]+2));
    sparsity_data correct_sd(2,contr_keys);

    if(contr_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_data::contract(...) returned incorrect value");
    }
}

void sparsity_data_test::test_contract_3d_1() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_contract_3d_1()";

    three_d_test_f tf = three_d_test_f();
    sparsity_data contr_sd = tf.sd.contract(1);

    size_t contr_keys_arr[20][2] = {{1,1},
                                    {1,3},
                                    {1,7},
                                    {1,9},
                                    {2,1},
                                    {2,2},
                                    {2,3},
                                    {2,4},
                                    {2,5},
                                    {4,2},
                                    {4,4},
                                    {4,5},
                                    {4,6},
                                    {4,7},
                                    {5,4},
                                    {5,6},
                                    {5,7},
                                    {7,5},
                                    {7,6},
                                    {7,7}};

    vector<idx_list> contr_keys;
    for(size_t key_idx = 0; key_idx < 20; ++key_idx)
        contr_keys.push_back(idx_list(contr_keys_arr[key_idx],contr_keys_arr[key_idx]+2));
    sparsity_data correct_sd(2,contr_keys);

    if(contr_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_data::contract(...) returned incorrect value");
    }
}

void sparsity_data_test::test_contract_3d_2() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_contract_3d_2()";

    three_d_test_f tf = three_d_test_f();
    sparsity_data contr_sd = tf.sd.contract(2);

    size_t contr_keys_arr[13][2] = {{1,2},
                                    {1,3},
                                    {1,5},
                                    {2,3},
                                    {2,4},
                                    {2,6},
                                    {4,1},
                                    {4,2},
                                    {4,3},
                                    {5,1},
                                    {5,2},
                                    {7,4},
                                    {7,7}};

    vector<idx_list> contr_keys;
    for(size_t key_idx = 0; key_idx < 13; ++key_idx)
        contr_keys.push_back(idx_list(contr_keys_arr[key_idx],contr_keys_arr[key_idx]+2));
    sparsity_data correct_sd(2,contr_keys);

    if(contr_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparsity_data::contract(...) returned incorrect value");
    }
}

void sparsity_data_test::test_fuse_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_fuse_3d_2d()";

    three_d_test_f tf_3d =  three_d_test_f();
    two_d_test_f tf_2d = two_d_test_f();

    sparsity_data fused_sd = tf_3d.sd.fuse(tf_2d.sd,idx_list(1,2),idx_list(1,0)); 

    size_t fused_keys_arr[23][4] = {{1,2,7,4},
                                    {1,3,1,2},
                                    {1,3,1,5},
                                    {2,3,1,2},
                                    {2,3,1,5},
                                    {2,4,2,3},
                                    {2,4,5,1},
                                    {2,4,5,2},
                                    {2,6,4,1},
                                    {2,6,4,4},
                                    {4,1,4,1},
                                    {4,1,4,4},
                                    {4,1,7,4},
                                    {4,2,2,3},
                                    {4,3,5,1},
                                    {4,3,5,2},
                                    {4,3,7,4},
                                    {5,1,4,1},
                                    {5,1,4,4},
                                    {5,2,7,4},
                                    {7,4,5,1},
                                    {7,4,5,2},
                                    {7,7,7,4}};

    vector<idx_list> fused_keys;
    for(size_t key_idx = 0; key_idx < 23; ++key_idx)
        fused_keys.push_back(idx_list(fused_keys_arr[key_idx],fused_keys_arr[key_idx]+4));

    size_t fused_vals_arr[23][2] = {{8,28},
                                    {16,0},
                                    {16,4},
                                    {32,0},
                                    {32,4},
                                    {40,8},
                                    {48,20},
                                    {48,24},
                                    {64,12},
                                    {64,16},
                                    {72,12},
                                    {72,16},
                                    {80,28},
                                    {88,8},
                                    {96,20},
                                    {96,24},
                                    {112,28},
                                    {120,12},
                                    {120,16},
                                    {136,28},
                                    {144,20},
                                    {144,24},
                                    {160,28}};

    sparsity_data correct_sd(4,fused_keys); 
    for(sparsity_data::iterator it = correct_sd.begin(); it != correct_sd.end(); ++it)
    {
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][0]);
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][1]);
    }

    if(fused_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "fuse returned incorrect sparsity_data");
    }
}

void sparsity_data_test::test_fuse_3d_3d_non_contiguous() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_fuse_3d_3d_non_contiguous()";

    //Work with a reduced subset of the keys to keep the result manageable
    three_d_test_f tf = three_d_test_f();
    vector<idx_list> keys,vals;
    for(sparsity_data::iterator it = tf.sd.begin(); it != tf.sd.begin()+11; ++it)
    {
        keys.push_back(it->first);
        vals.push_back(it->second);
    }
    sparsity_data sd_0(3,keys);
    sparsity_data sd_1(3,keys);

    for(sparsity_data::iterator it = sd_0.begin(); it != sd_0.end(); ++it) it->second = vals[distance(sd_0.begin(),it)];
    for(sparsity_data::iterator it = sd_1.begin(); it != sd_1.end(); ++it) it->second = vals[distance(sd_1.begin(),it)];

    sparsity_data fused_sd = sd_0.fuse(sd_1,idx_list(1,2),idx_list(1,1)); 

    size_t fused_keys_arr[15][5] = {{1,2,3,1,1},
                                    {1,2,3,2,1},
                                    {1,3,1,4,4},
                                    {1,3,1,4,7},
                                    {2,3,1,4,4},
                                    {2,3,1,4,7},
                                    {2,4,2,1,3},
                                    {2,4,2,1,7},
                                    {2,4,5,1,9},
                                    {2,6,3,1,1},
                                    {2,6,3,2,1},
                                    {2,6,4,2,2},
                                    {2,6,4,2,5},
                                    {4,1,4,2,2},
                                    {4,1,4,2,5}};

    vector<idx_list> fused_keys;
    for(size_t key_idx = 0; key_idx < 15; ++key_idx)
        fused_keys.push_back(idx_list(fused_keys_arr[key_idx],fused_keys_arr[key_idx]+5));

    size_t fused_vals_arr[15][2] = {{0,16},
                                    {0,32},
                                    {16,72},
                                    {16,80},
                                    {32,72},
                                    {32,80},
                                    {40,0},
                                    {40,8},
                                    {48,24},
                                    {56,16},
                                    {56,32},
                                    {64,40},
                                    {64,48},
                                    {72,40},
                                    {72,48}};

    sparsity_data correct_sd(5,fused_keys); 
    for(sparsity_data::iterator it = correct_sd.begin(); it != correct_sd.end(); ++it)
    {
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][0]);
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][1]);
    }

    if(fused_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "fuse returned incorrect sparsity_data");
    }
}

//ijk fused to jkl 
void sparsity_data_test::test_fuse_3d_3d_multi_index() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_fuse_3d_3d_multi_index";

    three_d_test_f tf = three_d_test_f();

    idx_list lhs_inds(1,1);
    lhs_inds.push_back(2);
    idx_list rhs_inds(1,0);
    rhs_inds.push_back(1);
    sparsity_data fused_sd = tf.sd.fuse(tf.sd,lhs_inds,rhs_inds); 

    size_t fused_keys_arr[5][4] = {{1,2,3,1},
                                   {2,4,2,2},
                                   {5,2,6,3},
                                   {5,2,6,4},
                                   {7,7,7,7}};

    vector<idx_list> fused_keys;
    for(size_t key_idx = 0; key_idx < 5; ++key_idx)
        fused_keys.push_back(idx_list(fused_keys_arr[key_idx],fused_keys_arr[key_idx]+4));

    size_t fused_vals_arr[5][2] = {{0,32},
                                   {40,88},
                                   {128,56},
                                   {128,64},
                                   {160,160}};

    sparsity_data correct_sd(4,fused_keys); 
    for(sparsity_data::iterator it = correct_sd.begin(); it != correct_sd.end(); ++it)
    {
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][0]);
        it->second.push_back(fused_vals_arr[distance(correct_sd.begin(),it)][1]);
    }

    if(fused_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "fuse returned incorrect sparsity_data");
    }
}

void sparsity_data_test::test_truncate_subspace_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_truncate_subspace_3d()";

    three_d_test_f tf = three_d_test_f();

    idx_pair subspace_bounds(3,5);
    sparsity_data trunc_sd = tf.sd.truncate_subspace(1,subspace_bounds);

    size_t trunc_keys_arr[9][3] = {{1,3,1},
                                   {2,3,1},
                                   {2,4,2},
                                   {2,4,5},
                                   {4,3,5},
                                   {4,3,6},
                                   {4,3,7},
                                   {7,4,5},
                                   {7,4,6}};

    vector<idx_list> trunc_keys;
    for(size_t key_idx = 0; key_idx < 9; ++key_idx)
        trunc_keys.push_back(idx_list(trunc_keys_arr[key_idx],trunc_keys_arr[key_idx]+3));

    size_t trunc_vals_arr[9] = {16,32,40,48,96,104,112,144,152};

    sparsity_data correct_sd(3,trunc_keys); 
    for(sparsity_data::iterator it = correct_sd.begin(); it != correct_sd.end(); ++it)
    {
        it->second.push_back(trunc_vals_arr[distance(correct_sd.begin(),it)]);
    }

    if(trunc_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "truncate_subspace returned incorrect sparsity_data");
    }
}

void sparsity_data_test::test_insert_subspace_3d() throw(libtest::test_exception)
{
    static const char *test_name = "sparsity_data_test::test_insert_subspace_3d()";

    three_d_test_f tf = three_d_test_f();
    sparsity_data insert_sd = tf.sd.insert_entries(2,range(0,2));

    size_t insert_keys_arr[42][4] = {{1,2,0,3},
                                     {1,2,0,7},
                                     {1,2,1,3},
                                     {1,2,1,7},
                                     {1,3,0,1},
                                     {1,3,1,1},
                                     {1,5,0,9},
                                     {1,5,1,9},
                                     {2,3,0,1},
                                     {2,3,1,1},
                                     {2,4,0,2},
                                     {2,4,0,5},
                                     {2,4,1,2},
                                     {2,4,1,5},
                                     {2,6,0,3},
                                     {2,6,0,4},
                                     {2,6,1,3},
                                     {2,6,1,4},
                                     {4,1,0,4},
                                     {4,1,0,7},
                                     {4,1,1,4},
                                     {4,1,1,7},
                                     {4,2,0,2},
                                     {4,2,1,2},
                                     {4,3,0,5},
                                     {4,3,0,6},
                                     {4,3,0,7},
                                     {4,3,1,5},
                                     {4,3,1,6},
                                     {4,3,1,7},
                                     {5,1,0,4},
                                     {5,1,1,4},
                                     {5,2,0,6},
                                     {5,2,0,7},
                                     {5,2,1,6},
                                     {5,2,1,7},
                                     {7,4,0,5},
                                     {7,4,0,6},
                                     {7,4,1,5},
                                     {7,4,1,6},
                                     {7,7,0,7},
                                     {7,7,1,7}};

    vector<idx_list> insert_keys;
    for(size_t key_idx = 0; key_idx < 42; ++key_idx)
        insert_keys.push_back(idx_list(insert_keys_arr[key_idx],insert_keys_arr[key_idx]+4));

    sparsity_data correct_sd(4,insert_keys); 
    if(insert_sd != correct_sd)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "insert_subspace returned incorrect sparsity_data");
    }
}

} // namespace libtensor

