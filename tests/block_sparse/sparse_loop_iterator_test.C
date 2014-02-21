/* * sparse_loop_list_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */

#include <libtensor/block_sparse/sparse_loop_iterator.h>
#include "sparse_loop_iterator_test.h"

using namespace std;

namespace libtensor {

namespace {

//Pretend we are iterating over 'i' in C_ij = A_ik B_kj
class dense_test_f {
private:
    static std::vector< sparse_bispace<1> > init_bispaces()
    {
        sparse_bispace<1> spb_i(12);
        vector<size_t> split_points_i;
        split_points_i.push_back(3);
        split_points_i.push_back(7);
        spb_i.split(split_points_i);

        sparse_bispace<1> spb_j(8);
        vector<size_t> split_points_j;
        split_points_j.push_back(2);
        spb_j.split(split_points_j);

        sparse_bispace<1> spb_k(15);
        vector<size_t> split_points_k;
        split_points_k.push_back(7);
        split_points_k.push_back(9);
        split_points_k.push_back(12);
        spb_k.split(split_points_k);

        std::vector< sparse_bispace<1> > bispaces(1,spb_i);
        bispaces.push_back(spb_j);
        bispaces.push_back(spb_k);
        return bispaces;
    }
public:
    sparse_bispace<2> spb_C;
    sparse_bispace<2> spb_A;
    sparse_bispace<2> spb_B;

    vector< pair<size_t,size_t> > dense_bispaces_and_index_groups;

    dense_test_f() : spb_C(init_bispaces()[0]|init_bispaces()[1]),
                     spb_A(init_bispaces()[0]|init_bispaces()[2]),
                     spb_B(init_bispaces()[2]|init_bispaces()[1])
    {
        dense_bispaces_and_index_groups.push_back(size_t_pair(0,0));
        dense_bispaces_and_index_groups.push_back(size_t_pair(1,0));
    }
};

//Same as above only now we make A sparse
//bispaces_and_index_groups can remain the same because we chose i loop
class sparse_test_f : public dense_test_f {
private:
    static std::vector< sequence<2,size_t> > init_block_tuples_list()
    {
        size_t seq0_arr[2] = {0,4};
        size_t seq1_arr[2] = {1,3};
        size_t seq2_arr[2] = {2,1};
        size_t seq3_arr[2] = {2,4};
        
        vector< sequence<2,size_t> > block_tuples_list(4);
        for(size_t i = 0; i < 2; ++i) block_tuples_list[0][i] = seq0_arr[i];
        for(size_t i = 0; i < 2; ++i) block_tuples_list[1][i] = seq1_arr[i];
        for(size_t i = 0; i < 2; ++i) block_tuples_list[2][i] = seq2_arr[i];
        for(size_t i = 0; i < 2; ++i) block_tuples_list[3][i] = seq3_arr[i];

        return block_tuples_list;
    }
    static std::vector< sparse_bispace<1> > init_subspaces()
    {
        dense_test_f base = dense_test_f();
        std::vector< sparse_bispace<1> > subspaces(1,base.spb_A[0]);
        subspaces.push_back(base.spb_A[1]);
        return subspaces;
    }
public:
    vector< size_t_pair > sparse_bispaces_and_index_groups;
    sparse_block_tree<2>  sbt_A;

    sparse_test_f() : sbt_A(init_block_tuples_list(),init_subspaces())
    {
        //The 'A' entry is now sparse,replace with the 'B' entry for 'k'
        dense_bispaces_and_index_groups.back() = size_t_pair(2,0);
        sparse_bispaces_and_index_groups.push_back(size_t_pair(1,0));

    }
};

} // namespace unnamed

void sparse_loop_iterator_test::perform() throw(libtest::test_exception) {

    test_set_offsets_and_dims_dense();
    test_increment_dense();

    /*test_set_offsets_and_dims_sparse();*/
}

void sparse_loop_iterator_test::test_set_offsets_and_dims_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_list_test::test_set_offsets_and_dims_dense()";

    dense_test_f tf = dense_test_f();

    //Initialize all untouched entries to a placeholder
    vector< offset_list > offset_lists(3,offset_list(2,1337));
    vector< dim_list > dim_lists(3,dim_list(2,1337));

    sparse_loop_iterator sli(tf.spb_A[0],tf.dense_bispaces_and_index_groups);
    sli.set_offsets_and_dims(offset_lists,dim_lists);

    //Benchmark values
    vector< offset_list > correct_offset_lists(offset_lists);
    correct_offset_lists[0][0] = 0;
    correct_offset_lists[1][0] = 0;
    vector< dim_list > correct_dim_lists(dim_lists);
    correct_dim_lists[0][0] = 3;
    correct_dim_lists[1][0] = 3;

    if(offset_lists != correct_offset_lists)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect offset_lists");
    }
    if(dim_lists != correct_dim_lists)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect dim_lists");
    }
}

void sparse_loop_iterator_test::test_increment_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_list_test::test_increment_dense()";

    dense_test_f tf = dense_test_f();

    //Initialize all untouched entries to a placeholder
    vector< offset_list > offset_lists(3,offset_list(2,1337));
    vector< dim_list > dim_lists(3,dim_list(2,1337));

    //Benchmark values
    vector< vector< offset_list > > correct_offset_lists_sets(3,vector< offset_list>(3,offset_list(2,1337)));
    vector< vector< dim_list > > correct_dim_lists_sets(3,vector< dim_list>(3,dim_list(2,1337)));
    for(size_t i = 0; i < tf.spb_A[0].get_n_blocks(); ++i)
    {
        size_t offset = tf.spb_A[0].get_block_abs_index(i);
        size_t dim = tf.spb_A[0].get_block_size(i);
        correct_offset_lists_sets[i][0][0] = offset; 
        correct_offset_lists_sets[i][1][0] = offset; 
        correct_dim_lists_sets[i][0][0] = dim; 
        correct_dim_lists_sets[i][1][0] = dim; 
    }

    size_t i = 0;
    for(sparse_loop_iterator sli(tf.spb_A[0],tf.dense_bispaces_and_index_groups); !sli.done(); ++sli)
    {
        const vector< offset_list >& correct_offset_lists =  correct_offset_lists_sets[i];
        const vector< dim_list >& correct_dim_lists =  correct_dim_lists_sets[i];

        sli.set_offsets_and_dims(offset_lists,dim_lists);
        if(offset_lists != correct_offset_lists)
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect offset_lists after increment");
        }
        if(dim_lists != correct_dim_lists)
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect dim_lists after increment");
        }
        ++i;
    }
    if(i != (correct_offset_lists_sets.size() - 1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_loop_iterator::operator++(...) returned end too early");
    }
}

#if 0
void sparse_loop_iterator_test::test_set_offsets_and_dims_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_loop_list_test::test_set_offsets_and_dims_sparse()";

    sparse_test_f tf = sparse_test_f();
    std::vector< sparse_bispace<1> > tree_subspaces(1,tf.spb_A[0]);
    tree_subspaces.push_back(tf.spb_A[1]);

    //Initialize all untouched entries to a placeholder
    vector< offset_list > offset_lists(3,offset_list(2,1337));
    offset_lists[1].pop_back();
    vector< dim_list > dim_lists(3,dim_list(2,1337));
    dim_lists[1].pop_back();

    sparse_loop_iterator sli(tf.sbt_A,tree_subspaces,tf.dense_bispaces_and_index_groups,tf.sparse_bispaces_and_index_groups);
    sli.set_offsets_and_dims(offset_lists,dim_lists);

    //Benchmark values
    vector< offset_list > correct_offset_lists(offset_lists);
    correct_offset_lists[0][0] = 0;
    correct_offset_lists[1][0] = 0;
    correct_offset_lists[2][0] = 12;
    vector< dim_list > correct_dim_lists(dim_lists);
    correct_dim_lists[0][0] = 3;
    correct_dim_lists[1][0] = 3;
    correct_dim_lists[2][0] = 3;

    if(offset_lists != correct_offset_lists)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect offset_lists");
    }
    if(dim_lists != correct_dim_lists)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_loop_iterator::set_offsets_and_dims(...) returned incorrect dim_lists");
    }
}
#endif

} // namespace libtensor
