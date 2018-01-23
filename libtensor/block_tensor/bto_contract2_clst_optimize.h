#ifndef LIBTENSOR_BTO_CONTRACT2_CLST_OPTIMIZE_H
#define LIBTENSOR_BTO_CONTRACT2_CLST_OPTIMIZE_H

#include <libtensor/core/contraction2.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2_clst.h>

namespace libtensor {


/** \brief Optimizes the contraction block list

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_clst_optimize {
public:
    typedef typename gen_bto_contract2_clst<N, M, K, T>::list_type
        contr_list;
    typedef typename contr_list::iterator iterator;

private:
    contraction2<N, M, K> m_contr;

public:
    bto_contract2_clst_optimize(const contraction2<N, M, K> &contr) :
        m_contr(contr)
    { }

    void perform(contr_list &clst);

private:
    bool check_same_blocks(const iterator &i1, const iterator &i2);
    bool check_same_contr(const contraction2<N, M, K> &contr1,
        const contraction2<N, M, K> &contr2);

};

template<size_t N, size_t M, size_t K>
using btod_contract2_clst_optimize = bto_contract2_clst_optimize<N, M, K, double>; 

} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_CLST_OPTIMIZE_H
