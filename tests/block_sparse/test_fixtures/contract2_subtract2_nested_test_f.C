#include "contract2_subtract2_nested_test_f.h" 
namespace libtensor {

const size_t contract2_subtract2_nested_test_f::ml_sparsity[4][2] = {{0,1},
                                                    {0,2},
                                                    {1,0},
                                                    {1,2}};

sparse_bispace<1> contract2_subtract2_nested_test_f::init_m()
{
    //Bispace for m
    sparse_bispace<1> spb_m(6);
    std::vector<size_t> split_points_m;
    split_points_m.push_back(3);
    spb_m.split(split_points_m);
    return spb_m;
}


const double contract2_subtract2_nested_test_f::s_F_arr[18] = { //i = 0 l = 0
                                                                1,2,
                                                                
                                                                //i = 0 l = 1
                                                                3,4,5,

                                                                //i = 0 l = 2
                                                                6,

                                                                //i = 1 l = 0
                                                                7,8,
                                                                9,10,
                                                                
                                                                //i = 1 l = 1
                                                                11,12,13,
                                                                14,15,16,
                                                                
                                                                //i = 1 l = 2
                                                                17,
                                                                18};
                                                            

const double contract2_subtract2_nested_test_f::s_D_arr[21] = {  //m = 0 l = 1
                                                                 1,2,3,
                                                                 4,5,6,
                                                                 7,8,9,

                                                                 //m = 0 l = 2
                                                                 10,
                                                                 11,
                                                                 12,

                                                                 //m = 1 l = 0
                                                                 13,14,
                                                                 15,16,
                                                                 17,18,

                                                                 //m = 1 l = 2
                                                                 19,
                                                                 20,
                                                                 21};


const double contract2_subtract2_nested_test_f::s_E_arr[18] = { //m = 0 i = 0
                                                                21926,47151,72376,

                                                                //m = 0 i = 1
                                                                122364,133052, 
                                                                240525,260660,
                                                                358686,388268,

                                                                //m = 1 i = 0
                                                                55172,
                                                                62381,
                                                                69590,

                                                                //m = 1 i = 1
                                                                302222,329427,
                                                                338618,369008,
                                                                375014,408589};

} // namespace libtensor
