#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_to_add_space.h>
#include <libtensor/diag_tensor/diag_tod_adjust_space.h>
#include <libtensor/diag_tensor/diag_tod_set.h>
#include "diag_tod_aux_constr_add.h"
#include "../diag_tod_copy.h"

namespace libtensor {


template<size_t N>
const char *diag_tod_copy<N>::k_clazz = "diag_tod_copy<N>";


template<size_t N>
void diag_tod_copy<N>::perform(bool zero, diag_tensor_wr_i<N, double> &dtb) {

    diag_tod_copy<N>::start_timer();

    try {

        const permutation<N> &perma = m_tra.get_perm();
        double ka = m_tra.get_scalar_tr().get_coeff();

        diag_tensor_rd_ctrl<N, double> ca(m_dta);
        diag_tensor_wr_ctrl<N, double> cb(dtb);

        //  Prepare the output space

        {
            diag_tensor_space<N> dtsb0(m_dta.get_space());
            dtsb0.permute(perma);
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
            ssb0.permute(perma);

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
                    diag_tod_aux_constr_add<N>(dtsa.get_dims(), ssa, pa,
                        dtsa.get_subspace_size(ssla[ssia]), m_tra).
                        perform(ssb, pb, dtsb.get_subspace_size(sslb[ssib]));
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


} // namespace libtensor

