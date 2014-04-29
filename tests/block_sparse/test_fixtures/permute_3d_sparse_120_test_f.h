#include <libtensor/block_sparse/sparse_bispace.h>
#include <string.h>

namespace libtensor {

class permute_3d_sparse_120_test_f  
{
private:
    static sparse_bispace<3> init_input_bispace();
    static permutation<3>  init_perm();
public:
    //3x4x5
    //Permutation is kij -> ijk 
	//Indices in comments are block indices
    static const double s_input_arr[35];
    static const double s_output_arr[35];

    double input_arr[35];
    double output_arr[35];

    permutation<3> perm;
    sparse_bispace<3> input_bispace;
    sparse_bispace<3> output_bispace;

    permute_3d_sparse_120_test_f() : perm(init_perm()),input_bispace(init_input_bispace()),output_bispace(init_input_bispace().permute(init_perm()))
    {
        memcpy(input_arr,s_input_arr,35*sizeof(double));
        memcpy(output_arr,s_output_arr,35*sizeof(double));
    }
};

} // namespace libtensor
