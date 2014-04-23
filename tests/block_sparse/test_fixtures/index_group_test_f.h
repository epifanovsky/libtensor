#ifndef INDEX_GROUP_TEST_F_H
#define INDEX_GROUP_TEST_F_H

#include <libtensor/block_sparse/sparse_bispace.h>

namespace libtensor {

class index_groups_test_f {
private:
    static sparse_bispace<7> init_bispace()
    {
        //Need 5 blocks of size 2 
        sparse_bispace<1> spb_0(10);
        idx_list split_points_0;
        for(size_t i = 2; i < spb_0.get_dim(); i += 2)
        {
            split_points_0.push_back(i);
        } 
        spb_0.split(split_points_0);


        //Need 6 blocks of size 3
        sparse_bispace<1> spb_1(18);
        idx_list split_points_1;
        for(size_t i = 3; i < spb_1.get_dim(); i += 3)
        {
            split_points_1.push_back(i);
        }
        spb_1.split(split_points_1);

        size_t key_0_arr_0[2] = {2,1}; //offset 0
        size_t key_1_arr_0[2] = {3,2}; //offset 6
        size_t key_2_arr_0[2] = {4,3}; //offset 12
        size_t key_3_arr_0[2] = {5,4}; //offset 18
        std::vector< sequence<2,size_t> > sig_blocks_0(4);
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[0][i] = key_0_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[1][i] = key_1_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[2][i] = key_2_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[3][i] = key_3_arr_0[i];

        size_t key_0_arr_1[2] = {1,4}; //offset 0
        size_t key_1_arr_1[2] = {2,1}; //offset 9
        size_t key_2_arr_1[2] = {3,3}; //offset 18
        std::vector< sequence<2,size_t> > sig_blocks_1(3);
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[0][i] = key_0_arr_1[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[1][i] = key_1_arr_1[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[2][i] = key_2_arr_1[i];

        return spb_0 | spb_0 | spb_1 % spb_0 << sig_blocks_0 | spb_0 | spb_1 % spb_1 << sig_blocks_1;
    }
public:
    sparse_bispace<7> bispace;
    index_groups_test_f() : bispace(init_bispace()) {}
};

} //namespace libtensor 

#endif /* INDEX_GROUP_TEST_F_H */
