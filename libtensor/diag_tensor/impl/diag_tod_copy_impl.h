#include <list>
#include <vector>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dadd2.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_to_add_space.h>
#include <libtensor/diag_tensor/diag_tod_adjust_space.h>
#include <libtensor/diag_tensor/diag_tod_set.h>
#include "../diag_tod_copy.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_copy<N>::k_clazz = "diag_tod_copy<N>";


template<size_t N>
void diag_tod_copy<N>::perform(bool zero, double c,
    diag_tensor_wr_i<N, double> &dtb) {

    diag_tod_copy<N>::start_timer();

    try {

        diag_tensor_rd_ctrl<N, double> ca(m_dta);
        diag_tensor_wr_ctrl<N, double> cb(dtb);

        //  Prepare the output space

        {
            diag_tensor_space<N> dtsb0(m_dta.get_space());
            dtsb0.permute(m_perma);
            if(zero) cb.req_remove_all_subspaces();
            diag_to_add_space<N> dtsb(dtb.get_space(), dtsb0);
            diag_tod_adjust_space<N>(dtsb.get_dtsc()).perform(dtb);
        }

        //  For each subspace in the source tensor, find a matching
        //  subspace in the destination tensor

        const diag_tensor_space<N> &dtsa = m_dta.get_space();
        const diag_tensor_space<N> &dtsb = dtb.get_space();

        std::vector<size_t> ssla, sslb;
        dtsa.get_all_subspaces(ssla);
        dtsb.get_all_subspaces(sslb);

        for(size_t ssia = 0; ssia < ssla.size(); ssia++) {

            const diag_tensor_subspace<N> &ssa = dtsa.get_subspace(ssla[ssia]);
            diag_tensor_subspace<N> ssb0(ssa);
            ssb0.permute(m_perma);

            bool match_found = false;
            for(size_t ssib = 0; !match_found && ssib < sslb.size(); ssib++) {
                const diag_tensor_subspace<N> &ssb =
                    dtsb.get_subspace(sslb[ssib]);
                bool match = true;
                for(size_t i = 0; match && i < N; i++) {
                    mask<N> ma, mb;
                    for(size_t ia = 0; ia < ssb0.get_ndiag(); ia++) {
                        const mask<N> &mda = ssb0.get_diag_mask(ia);
                        if(mda[i]) ma |= mda;
                    }
                    for(size_t ib = 0; ib < ssb.get_ndiag(); ib++) {
                        const mask<N> &mdb = ssb.get_diag_mask(ib);
                        if(mdb[i]) mb |= mdb;
                    }
                    if(!ma[i]) ma[i] = true;
                    if(!mb[i]) mb[i] = true;
                    mask<N> ma2(ma);
                    ma2 |= mb;
                    if(!ma2.equals(ma)) match = false;
                }
                if(match) {
                    match_found = true;
                    const double *pa = ca.req_const_dataptr(ssla[ssia]);
                    double *pb = cb.req_dataptr(sslb[ssib]);
                    constrained_copy(dtsa.get_dims(), ssa, pa,
                        dtsa.get_subspace_size(ssla[ssia]), m_perma, m_ka * c,
                        ssb, pb, dtsb.get_subspace_size(sslb[ssib]));
                    cb.ret_dataptr(sslb[ssib], pb);
                    ca.ret_const_dataptr(ssla[ssia], pa);
                }
            }
        }

    } catch(...) {
        diag_tod_copy<N>::stop_timer();
        throw;
    }

    diag_tod_copy<N>::stop_timer();
}


template<size_t N>
void diag_tod_copy<N>::constrained_copy(const dimensions<N> &dims,
    const diag_tensor_subspace<N> &ss1, const double *p1, size_t sz1,
    const permutation<N> &perm, double d, const diag_tensor_subspace<N> &ss2,
    double *p2, size_t sz2) {

    std::list< loop_list_node<2, 1> > lpadd1, lpadd2;
    typename std::list< loop_list_node<2, 1> >::iterator iadd = lpadd1.end();

    double zero = 0.0;

    dimensions<N> dims1(dims), dims2(dims);
    dims2.permute(perm);

    mask<N> mdone;
    for(size_t i = 0; i < N; i++) if(!mdone[i]) {

        mask<N> m01, m02, m1, m1p, m2;
        m01[i] = true;
        m02[i] = true;
        m02.permute(perm);
        do {
            mark_diags(m01, ss1, m1);
            mark_diags(m02, ss2, m2);
            m01 |= m1;
            m02 |= m2;
            m1p = m1;
            m1p.permute(perm);
        } while(!m1p.equals(m2));

        iadd = lpadd1.insert(lpadd1.end(), loop_list_node<2, 1>(dims1[i]));
        size_t inc1 = get_increment(dims1, ss1, m01);
        size_t inc2 = get_increment(dims2, ss2, m02);
        iadd->stepa(0) = inc1;
        iadd->stepa(1) = 0;
        iadd->stepb(0) = inc2;

        mdone |= m01;
    }

    loop_registers<2, 1> radd;
    radd.m_ptra[0] = p1;
    radd.m_ptra[1] = &zero;
    radd.m_ptrb[0] = p2;
    radd.m_ptra_end[0] = p1 + sz1;
    radd.m_ptra_end[1] = &zero + 1;
    radd.m_ptrb_end[0] = p2 + sz2;

    {
        diag_tod_copy<N>::start_timer("copy");
        std::auto_ptr< kernel_base<linalg, 2, 1> > kern_add(
            kern_dadd2<linalg>::match(d, 1.0, 1.0, lpadd1, lpadd2));
        loop_list_runner<linalg, 2, 1>(lpadd1).run(0, radd, *kern_add);
        diag_tod_copy<N>::stop_timer("copy");
    }
}


template<size_t N>
void diag_tod_copy<N>::mark_diags(const mask<N> &m0,
    const diag_tensor_subspace<N> &ss, mask<N> &m1) {

    //  Input: one or more bits set in m0
    //  Output: for each bit set in m0, the respective diagonal is marked in m1

    size_t ndiag = ss.get_ndiag();
    for(size_t i = 0; i < ndiag; i++) {
        const mask<N> &m = ss.get_diag_mask(i);
        for(size_t j = 0; j < N; j++) if(m0[j] && m[j]) {
            m1 |= m;
            break;
        }
    }
    m1 |= m0;
}


template<size_t N>
size_t diag_tod_copy<N>::get_increment(const dimensions<N> &dims,
    const diag_tensor_subspace<N> &ss, const mask<N> &m) const {

    //  Build new dimensions in which only the primary index
    //  of each diagonal exists

    index<N> i1, i2;
    mask<N> mm;

    size_t ndiag = ss.get_ndiag(); // Total number of diagonals
    const mask<N> &totm = ss.get_total_mask();
    for(size_t i = 0; i < ndiag; i++) {
        const mask<N> &dm = ss.get_diag_mask(i);
        for(size_t j = 0; j < N; j++) if(dm[j]) {
            i2[j] = dims[j] - 1;
            if(m[j]) mm[j] = true;
            break;
        }
    }
    for(size_t j = 0; j < N; j++) if(!totm[j]) {
        i2[j] = dims[j] - 1;
        mm[j] = m[j];
    }

    dimensions<N> dims2(index_range<N>(i1, i2));

    //  Now compute and return increment

    size_t inc = 0;
    for(size_t j = 0; j < N; j++) if(mm[j]) inc += dims2.get_increment(j);
    return inc;
}


} // namespace libtensor

