/* * direct_eval_functor_test.C
 *
 *  Created on: Nov 13, 2013
 *      Author: smanzer
 */
#include "direct_eval_functor_test.h"

using namespace std;

namespace libtensor {

void direct_eval_functor_test::perform() throw(libtest::test_exception) {
    test_contract_then_subtract();
}

#if 0
//Stub should return the appropriate batches 
class C_stub : public labeled_batch_provider_i<double*>
{
private:
    double A_arr[12] = {3,7,9,4,
                        2,8,14,11,
                        12,20,15,8};
    letter_expr<2> m_le;
public:
    C_stub(const sparse_bispace_any_order& bispace,const letter_expr<2>& le) : m_le(le);

    get_batch(size_t batched_index,const idx_pair& batch_limits,double* batch_mem)
    {
        if(batched_index != 0)
        {
            throw bad_parameter(g_ns,"C_stub","get_batch(...)",
                    __FILE__, __LINE__, "invalid batched_index specified");
        }
        if(batch_limits == idx_pair(0,2))
        {
            for(size_t i = 0; i < 8; ++i) 
            {
                batch_mem[i] = A_arr[i]
            }
        }
        else if(batch_limits == idx_pair(2,3))
        {
            for(size_t i = 0; i < 4; ++i)
            {
                batch_mem[i] = A_arr[i+8]
            }
        }
    }

    const letter &letter_at(size_t i) const throw(out_of_bounds) {
        return m_le.letter_at(i);
    }

    bool contains(const letter &let) const {
        return m_le.contains(let);
    }

    size_t index_of(const letter &let) const throw(exception) {
        return m_le.index_of(let);
    }
}
#endif


//C = AB (direct)
//E = D - C
//We break the formation of C down into two batches, based on the i index
void direct_eval_functor_test::test_contract_then_subtract() throw(libtest::test_exception)
{
#if 0
    double A_arr[12] = {3,7,9,4,
                        2,8,14,11,
                        12,20,15,8};

    double B_arr[12] = {1,2,3,
                        4,5,6,
                        7,8,9,
                        10,11,12};

    //The direct intermediate C is: 
    //{134,157,180, 
    // 242,277,312,
    // 277,332,387}
    				
    double D_arr[9] = {13,14,15,
                       16,17,18,
                       19,20,21};

    double E_correct[9] = {121,143,165,
                           226,260,294,
                           258,312,366};


    sparse_bispace<1> spb_i(3);
    vector<size_t> split_points_i(1,2);
    spb_i.split(split_points_i);

    sparse_bispace<1> spb_k(4);
    vector<size_t> split_points_k(1,2);
    spb_k.split(split_points_k);

    sparse_bispace<1> spb_j(4);
    vector<size_t> split_points_j(1,1);
    spb_j.split(split_points_j);
    sparse_bispace<2> spb_C = spb_i | spb_j;

    sparse_bispace<2> spb_A  = spb_i | spb_k;
    sparse_bispace<2> spb_B  = spb_k | spb_j;
    

    sparse_btensor<2> A(spb_A,A_arr);
    sparse_btensor<2> B(spb_B,B_arr);
    direct_sparse_btensor<2> B(spb_B,B_arr);
    sparse_btensor<2> E(spb_E);

    letter i,j;
    direct_eval_functor def(C_stub(i|j),D(i|j),block_subtract2_kernel());
    def(E(i|j));
#endif
}

} // namespace libtensor
