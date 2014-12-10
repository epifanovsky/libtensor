#include "block_loop_test.h" 
#include <libtensor/block_sparse/block_loop.h>

using namespace std; 
namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_apply_contract2();
}

void block_loop_test::test_apply_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_apply_contract2()";

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
    loops.push_back(block_loop(sub_i,i_t_igs));

    idx_pair_list j_t_igs(1,idx_pair(0,1));
    j_t_igs.push_back(idx_pair(2,1));
    loops.push_back(block_loop(sub_j,j_t_igs));

    idx_pair_list k_t_igs(1,idx_pair(1,1));
    k_t_igs.push_back(idx_pair(2,0));
    loops.push_back(block_loop(sub_k,k_t_igs));

    vector<idx_list> orig_ig_offs(3,idx_list(2));
    orig_ig_offs[0][0] = 9;
    orig_ig_offs[0][1] = 1;
    orig_ig_offs[1][0] = 10;
    orig_ig_offs[1][1] = 1;
    orig_ig_offs[2][0] = 9;
    orig_ig_offs[2][1] = 1;

    vector<idx_list> ig_offs(orig_ig_offs);
    loops[0].apply(ig_offs);
    loops[1].apply(ig_offs);
    loops[2].apply(ig_offs);

    if(ig_offs != vector<idx_list>(3,idx_list(2,0)))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::apply() returned incorrect value");
    }

    ig_offs = orig_ig_offs;
    ++loops[2];
    loops[0].apply(ig_offs);
    loops[1].apply(ig_offs);
    loops[2].apply(ig_offs);
    vector<idx_list> c_ig_offs(3,idx_list(2));
    c_ig_offs[0][0] = 0;
    c_ig_offs[0][1] = 0;
    c_ig_offs[1][0] = 0;
    c_ig_offs[1][1] = 2;
    c_ig_offs[2][0] = 9;
    c_ig_offs[2][1] = 0;

    if(ig_offs != c_ig_offs)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::apply() returned incorrect value");
    }
}

} // namespace libtensor
