#ifndef LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_IMPL_H
#define LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_IMPL_H

#include <memory> // for auto_ptr
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd1.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "diag_tod_aux_constr_add.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_aux_constr_add<N>::k_clazz = "diag_tod_aux_constr_add<N>";


template<size_t N>
void diag_tod_aux_constr_add<N>::perform(
    const diag_tensor_subspace<N> &ssb,
    double *pb,
    size_t szb) {

    std::list< loop_list_node<1, 1> > lpadd1, lpadd2;
    typename std::list< loop_list_node<1, 1> >::iterator iadd = lpadd1.end();

    double d = m_tra.get_scalar_tr().get_coeff();

    permutation<N> perma(m_tra.get_perm()), pinva(perma, true);
    const dimensions<N> &dimsa = m_dimsa;
    dimensions<N> dimsb(dimsa);
    dimsb.permute(perma);

    mask<N> mdone;
    for(size_t i = 0; i < N; i++) if(!mdone[i]) {

        mask<N> m01, m02, m1, m1p, m2, m2p;
        m01[i] = true;
        m02[i] = true;
        m02.permute(perma);
        do {
            mark_diags(m01, m_ssa, m1);
            mark_diags(m02, ssb, m2);
            m01 |= m1;
            m02 |= m2;
            m1p = m1;
            m1p.permute(perma);
            m2p = m2;
            m2p.permute(pinva);
            m01 |= m2p;
            m02 |= m1p;
        } while(!m1p.equals(m2));

        iadd = lpadd1.insert(lpadd1.end(), loop_list_node<1, 1>(dimsa[i]));
        size_t inc1 = get_increment(dimsa, m_ssa, m01);
        size_t inc2 = get_increment(dimsb, ssb, m02);
        iadd->stepa(0) = inc1;
        iadd->stepb(0) = inc2;

        mdone |= m01;
    }

    loop_registers<1, 1> radd;
    radd.m_ptra[0] = m_pa;
    radd.m_ptrb[0] = pb;
    radd.m_ptra_end[0] = m_pa + m_sza;
    radd.m_ptrb_end[0] = pb + szb;

    {
        diag_tod_aux_constr_add::start_timer("copy");
        std::auto_ptr< kernel_base<linalg, 1, 1, double> > kern_add(
            kern_dadd1<linalg>::match(d, lpadd1, lpadd2));
        loop_list_runner<linalg, 1, 1>(lpadd1).run(0, radd, *kern_add);
        diag_tod_aux_constr_add::stop_timer("copy");
    }
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_AUX_CONSTR_ADD_IMPL_H
