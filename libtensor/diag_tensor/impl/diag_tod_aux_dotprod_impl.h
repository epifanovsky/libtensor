#include <memory> // for auto_ptr
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dmul2.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "diag_tod_aux_dotprod.h"

namespace libtensor {


template<size_t N>
const char diag_tod_aux_dotprod<N>::k_clazz[] = "diag_tod_aux_dotprod<N>";


template<size_t N>
double diag_tod_aux_dotprod<N>::calculate() {

    std::list< loop_list_node<2, 1> > lpmul1, lpmul2;
    typename std::list< loop_list_node<2, 1> >::iterator imul = lpmul1.end();

    double c = m_tra.get_scalar_tr().get_coeff() *
        m_trb.get_scalar_tr().get_coeff();

    permutation<N> perma(m_tra.get_perm()), pinva(perma, true);
    permutation<N> permb(m_trb.get_perm()), pinvb(permb, true);
    dimensions<N> dimsab(m_dimsa);
    dimsab.permute(perma);

    mask<N> mdone;
    for(size_t i = 0; i < N; i++) if(!mdone[i]) {

        mask<N> m01, m02, m1, m1p, m1t, m2, m2p, m2t;
        m01[i] = true;
        m02[i] = true;
        m01.permute(pinva);
        m02.permute(pinvb);
        do {
            mark_diags(m01, m_ssa, m1);
            mark_diags(m02, m_ssb, m2);
            m01 = m1;
            m02 = m2;
            m1t = m2;
            m2t = m1;
            m1t.permute(permb).permute(pinva);
            m2t.permute(perma).permute(pinvb);
            m01 |= m1t;
            m02 |= m2t;
            m1p = m1;
            m2p = m2;
            m1p.permute(perma);
            m2p.permute(permb);
        } while(!m1p.equals(m2p));

        imul = lpmul1.insert(lpmul1.end(), loop_list_node<2, 1>(dimsab[i]));
        imul->stepa(0) = get_increment(m_dimsa, m_ssa, m01);
        imul->stepa(1) = get_increment(m_dimsb, m_ssb, m02);
        imul->stepb(0) = 0;

        mdone |= m1p;
    }

    double d = 0.0;

    loop_registers<2, 1> rmul;
    rmul.m_ptra[0] = m_pa;
    rmul.m_ptra[1] = m_pb;
    rmul.m_ptrb[0] = &d;
    rmul.m_ptra_end[0] = m_pa + m_sza;
    rmul.m_ptra_end[1] = m_pb + m_szb;
    rmul.m_ptrb_end[0] = &d + 1;

    {
        std::auto_ptr< kernel_base<linalg, 2, 1, double> > kern(
            kern_dmul2<linalg>::match(1.0, lpmul1, lpmul2));
        diag_tod_aux_dotprod::start_timer(kern->get_name());
        loop_list_runner<linalg, 2, 1>(lpmul1).run(0, rmul, *kern);
        diag_tod_aux_dotprod::stop_timer(kern->get_name());
    }

    return d * c;
}


} // namespace libtensor

